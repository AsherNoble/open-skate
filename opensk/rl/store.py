"""Rollout storage, shaped so a sim episode and a real demonstration are one type.

A world model is trained here and fine-tuned on expert demonstrations captured
from the real game, so the two have to arrive in the same shape or the
fine-tuning stage becomes a translation exercise. They already do -- one
gesture, 68 frames over 2.3 s -- and this module's job is to keep it that way.

Stored as compressed `.npz` shards rather than one growing file: a 1024-wide
batch is the natural write unit (it is what the GPU produces in one call), and
shards can be written from several workers and read in any order.

Frames are stored as **uint8**, not float32: 4x smaller, and the renderer's
output is quantised to 8 bits per channel anyway, so the conversion is exact
rather than lossy. A 1024-episode shard of 68 frames at 128x64x3 is 1.7 GB in
float32 and 428 MB in uint8, which is the difference between a pipeline that
streams and one that does not.
"""
from __future__ import annotations

import json
import pathlib
import hashlib
import os
import tempfile
from dataclasses import dataclass

import numpy as np

# Bumped whenever the stored layout changes meaning. A reader that finds a
# version it does not know refuses the shard rather than misinterpreting it.
FORMAT_VERSION = 4


@dataclass(frozen=True)
class Shard:
    actions: np.ndarray       # (B, A) the flat gesture vectors that were run
    pos: np.ndarray           # (B, F, 3)
    quat: np.ndarray          # (B, F, 4)
    roll_deg: np.ndarray      # (B,)
    yaw_deg: np.ndarray       # (B,)
    peak_height: np.ndarray   # (B,)
    air_s: np.ndarray         # (B,)
    displacement: np.ndarray  # (B,)
    valid: np.ndarray         # (B,) bool
    source: str               # "sim" or "device"
    park: str                 # collision environment used for every episode
    appearance: str           # RGB preset used for every episode
    rgb: np.ndarray | None = None   # (B, F, H, W, 3) uint8, when rendered
    metadata: dict | None = None
    extras: dict | None = None

    def __len__(self) -> int:
        return len(self.actions)

    def filtered(self) -> "Shard":
        """Only the episodes that stayed physical.

        Kept as an explicit call rather than done at write time, because the
        proportion that fails is itself a measurement worth being able to make
        from stored data.
        """
        k = np.asarray(self.valid, dtype=bool)
        return Shard(*[np.asarray(getattr(self, n))[k] for n in _ARRAYS],
                     source=self.source, park=self.park,
                     appearance=self.appearance,
                     rgb=None if self.rgb is None else np.asarray(self.rgb)[k],
                     metadata=self.metadata,
                     extras={n: v[k] for n, v in (self.extras or {}).items()})

    def frames_float(self) -> np.ndarray:
        """Frames back as float32 in [0, 1], the form a model consumes."""
        if self.rgb is None:
            raise ValueError("this shard carries no frames")
        return np.asarray(self.rgb, dtype=np.float32) / 255.0


_ARRAYS = ("actions", "pos", "quat", "roll_deg", "yaw_deg", "peak_height",
           "air_s", "displacement", "valid")


def save(path, actions, episodes, source: str = "sim", *,
         park: str, appearance: str, metadata=None, extras=None) -> pathlib.Path:
    """Write one batch of episodes as a shard. Returns the path written."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"actions": np.asarray(actions, dtype=np.float32)}
    for name in _ARRAYS[1:]:
        v = np.asarray(getattr(episodes, name))
        data[name] = v.astype(bool if name == "valid" else np.float32)
    rgb = getattr(episodes, "rgb", None)
    if rgb is not None:
        # The renderer emits float in [0, 1] that was quantised to 8 bits on
        # the way out, so rounding back to uint8 loses nothing and saves 4x.
        rgb = np.asarray(rgb)
        data["rgb"] = (rgb if rgb.dtype == np.uint8 else
                       np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8))
    for name, value in (extras or {}).items():
        if name in data or name == "meta":
            raise ValueError(f"reserved array: {name}")
        data[name] = np.asarray(value)
    meta = dict(metadata or {})
    meta.update(version=FORMAT_VERSION, source=source, park=park,
                appearance=appearance,
                checksums={k: array_hash(v) for k, v in data.items()})
    data["meta"] = np.frombuffer(json.dumps(meta, sort_keys=True).encode(), np.uint8)
    validate(data, meta)
    with atomic_file(path) as stream:
        np.savez_compressed(stream, **data)
    return path


def load(path) -> Shard:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(bytes(z["meta"]).decode())
        if meta["version"] not in (1, 2, 3, FORMAT_VERSION):
            raise ValueError(
                f"{path}: format version {meta['version']}, expected "
                f"{FORMAT_VERSION}. Refusing to guess at the layout.")
        data = {k: z[k] for k in z.files}
    validate(data, meta)
    if meta["version"] < 3 and data["actions"].shape[1] % 9 == 8:
        # Old policy vectors had no rotate-button controls. Disabled gate;
        # leave the original values intact and never modify the source file.
        data["actions"] = np.pad(data["actions"], ((0, 0), (0, 3)))
        data["actions"][:, -3] = -1
    meta.setdefault("park", "unknown")
    meta.setdefault("appearance", "unknown")
    return Shard(*[data[n] for n in _ARRAYS], source=meta["source"],
                 park=meta["park"], appearance=meta["appearance"],
                 rgb=data.get("rgb"), metadata=meta,
                 extras={k: v for k, v in data.items()
                         if k not in (*_ARRAYS, "rgb", "meta")})


def iter_shards(directory):
    """Every shard under `directory`, in sorted order for reproducibility."""
    for p in sorted(pathlib.Path(directory).rglob("*.npz")):
        yield load(p)


def array_hash(a):
    a = np.asarray(a)
    return hashlib.sha256(str((a.dtype.str, a.shape)).encode()
                          + a.tobytes()).hexdigest()


def file_hash(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


from contextlib import contextmanager


@contextmanager
def atomic_file(path):
    """Same-filesystem rename is the commit point; a crash leaves .partial only."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial",
                               dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path, data):
    with atomic_file(path) as stream:
        stream.write(json.dumps(data, indent=2, sort_keys=True,
                                allow_nan=False).encode())


def validate(data, meta):
    if not isinstance(meta.get("source"), str):
        raise ValueError("missing source metadata")
    for key in _ARRAYS:
        if key not in data:
            raise ValueError(f"incomplete shard: missing {key}")
    actions, pos, quat = (data[k] for k in ("actions", "pos", "quat"))
    if actions.ndim != 2 or pos.ndim != 3 or pos.shape[-1] != 3:
        raise ValueError("invalid action/pose dimensions")
    b, f = pos.shape[:2]
    if not b or not f or len(actions) != b or quat.shape != (b, f, 4):
        raise ValueError("inconsistent episode/frame dimensions")
    for key in _ARRAYS[3:]:
        if data[key].shape != (b,):
            raise ValueError(f"invalid {key} dimensions")
    for key, value in data.items():
        if key == "meta":
            continue
        if value.dtype.hasobject or value.ndim == 0 or len(value) != b:
            raise ValueError(f"invalid array {key}")
    if not np.isfinite(actions).all():
        raise ValueError("nonfinite actions")
    if data['valid'].dtype != np.bool_:
        raise ValueError('valid must be boolean')
    for key in ('frame_times','frame_valid','frame_hashes','obstacle_pixels'):
        if key in data and data[key].shape != (b,f):
            raise ValueError(f'invalid {key} frame dimensions')
    if 'frame_times' in data:
        times=data['frame_times']
        if not np.isfinite(times).all() or np.any(np.diff(times,axis=1)<=0):
            raise ValueError('invalid/nonmonotonic frame times')
    for key in ('episode_id','group_id','outcome_known','variation_seed'):
        if key in data and data[key].shape != (b,):
            raise ValueError(f'invalid {key} dimensions')
    if 'episode_id' in data and len(set(data['episode_id'])) != b:
        raise ValueError('duplicate episode IDs inside shard')
    if 'action_features' in data:
        if data['action_features'].shape != (b,33) or not np.isfinite(data['action_features']).all():
            raise ValueError('invalid action features')
    labelled=data.get('outcome_known',np.ones(b,bool)) & data['valid']
    for key in _ARRAYS[1:-1]:
        if not np.isfinite(data[key][labelled]).all():
            raise ValueError(f'nonfinite labelled {key}')
    if "rgb" in data:
        rgb = data["rgb"]
        if rgb.ndim != 5 or rgb.shape[:2] != (b, f) or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError("invalid RGB dimensions or dtype")
        if 'deck_mask' in data and (data['deck_mask'].shape != rgb.shape[:-1]
                                     or data['deck_mask'].dtype != np.bool_):
            raise ValueError('invalid deck mask dimensions or dtype')
    if meta["version"] == FORMAT_VERSION:
        if not all(isinstance(meta.get(k), str) and meta[k] for k in ("park", "appearance")):
            raise ValueError("missing park/appearance metadata")
        expected = meta.get("checksums", {})
        if set(expected) != set(data) - {"meta"}:
            raise ValueError("missing array checksums")
        for key, digest in expected.items():
            if array_hash(data[key]) != digest:
                raise ValueError(f"checksum mismatch: {key}")


def migrate(source, destination):
    """Lossless array migration into a NEW path; original shard is retained."""
    source, destination = pathlib.Path(source), pathlib.Path(destination)
    if source.resolve() == destination.resolve() or destination.exists():
        raise ValueError("migration requires a new destination; retain original")
    shard = load(source)
    meta = dict(shard.metadata, migrated_from=str(source),
                original_sha256=file_hash(source), original_version=shard.metadata["version"])
    return save(destination, shard.actions, shard, source=shard.source,
                park=shard.park, appearance=shard.appearance,
                metadata=meta, extras=shard.extras)


def collect(env, n_episodes: int, *, batch: int = 1024, out=None,
            seed: int = 0, verbose: bool = True):
    """Run `n_episodes` through `env` and write them as shards.

    Batch defaults to 1024 because that is where A10G throughput peaks --
    B=4096 measured 3.3x slower, so a larger batch is not a bigger bite.
    """
    out = pathlib.Path(out or "data/rollouts")
    done, i, paths = 0, 0, []
    step_batch = env.batch if getattr(env, "pixels", False) else batch
    while done < n_episodes:
        n = min(step_batch, n_episodes - done)
        # Pixel render contexts have a construction-time world count.  Run a
        # complete final batch and truncate only what is saved; asking the env
        # to resize the last call fails after an expensive compilation.
        run_n = step_batch if getattr(env, "pixels", False) else n
        run_actions = env.sample_actions(run_n, seed=seed + i)
        target = out / f"shard_{i:05d}.npz"
        identity = dict(seed=seed+i, count=n, batch=run_n,
                        seconds=getattr(env, 'seconds', None))
        if target.exists():
            old = load(target)
            if (old.metadata.get('collection') != identity or old.park != env.park_name
                    or old.appearance != env.appearance or len(old) != n
                    or not np.array_equal(old.actions,run_actions[:n].astype(np.float32))):
                raise ValueError(f"resume mismatch: {target}; use a new output directory")
            paths.append(target)
            done += n
            i += 1
            continue
        run_episodes = env.step(run_actions)
        actions = run_actions[:n]
        episodes = type(run_episodes)(*(
            None if value is None else np.asarray(value)[:n]
            for value in run_episodes))
        paths.append(save(target, actions, episodes,
                          park=env.park_name, appearance=env.appearance,
                          metadata=dict(collection=identity)))
        kept = int(np.asarray(episodes.valid).sum())
        if verbose:
            print(f"shard {i}: {n} episodes, {kept} physical", flush=True)
        done += n
        i += 1
    return paths


# --- expert demonstrations ------------------------------------------------

def demo_actions(samples) -> np.ndarray:
    """Real captured gestures as the recipes the device executed.

    Returned as recipes rather than action vectors: the action space squashes
    through tanh, so an arbitrary recipe has no exact pre-image and inverting
    it would quietly move the demonstration. A demonstration is ground truth
    and must not be adjusted to fit our parameterisation -- when a policy has
    to be compared against one, compare in recipe space.
    """
    return [s.recipe() for s in samples]
