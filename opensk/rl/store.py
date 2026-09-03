"""Rollout storage, shaped so a sim episode and a real demonstration are one type.

A world model is trained here and fine-tuned on expert demonstrations captured
from the real game, so the two have to arrive in the same shape or the
fine-tuning stage becomes a translation exercise. They already do -- one
gesture, 68 frames over 2.3 s -- and this module's job is to keep it that way.

Stored as compressed `.npz` shards rather than one growing file: a 1024-wide
batch is the natural write unit (it is what the GPU produces in one call), and
shards can be written from several workers and read in any order.

Deliberately NOT storing pixels yet. Whether observations are RGB frames or
segmentation masks is still open -- it turns on the renderer throughput
measurement -- so this stores pose trajectories and actions, which both
observation choices are derived from and neither makes obsolete.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np

# Bumped whenever the stored layout changes meaning. A reader that finds a
# version it does not know refuses the shard rather than misinterpreting it.
FORMAT_VERSION = 1


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

    def __len__(self) -> int:
        return len(self.actions)

    def filtered(self) -> "Shard":
        """Only the episodes that stayed physical.

        Kept as an explicit call rather than done at write time, because the
        proportion that fails is itself a measurement worth being able to make
        from stored data.
        """
        k = np.asarray(self.valid, dtype=bool)
        return Shard(*[np.asarray(getattr(self, f.name))[k] for f in
                       self.__dataclass_fields__.values()
                       if f.name != "source"], source=self.source)


_ARRAYS = ("actions", "pos", "quat", "roll_deg", "yaw_deg", "peak_height",
           "air_s", "displacement", "valid")


def save(path, actions, episodes, source: str = "sim") -> pathlib.Path:
    """Write one batch of episodes as a shard. Returns the path written."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"actions": np.asarray(actions, dtype=np.float32)}
    for name in _ARRAYS[1:]:
        v = np.asarray(getattr(episodes, name))
        data[name] = v.astype(bool if name == "valid" else np.float32)
    data["meta"] = np.frombuffer(
        json.dumps({"version": FORMAT_VERSION, "source": source}).encode(),
        dtype=np.uint8)
    np.savez_compressed(path, **data)
    return path


def load(path) -> Shard:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(bytes(z["meta"]).decode())
        if meta["version"] != FORMAT_VERSION:
            raise ValueError(
                f"{path}: format version {meta['version']}, expected "
                f"{FORMAT_VERSION}. Refusing to guess at the layout.")
        return Shard(*[z[n] for n in _ARRAYS], source=meta["source"])


def iter_shards(directory):
    """Every shard under `directory`, in sorted order for reproducibility."""
    for p in sorted(pathlib.Path(directory).glob("*.npz")):
        yield load(p)


def collect(env, n_episodes: int, *, batch: int = 1024, out=None,
            seed: int = 0, verbose: bool = True):
    """Run `n_episodes` through `env` and write them as shards.

    Batch defaults to 1024 because that is where A10G throughput peaks --
    B=4096 measured 3.3x slower, so a larger batch is not a bigger bite.
    """
    out = pathlib.Path(out or "data/rollouts")
    done, i, paths = 0, 0, []
    while done < n_episodes:
        n = min(batch, n_episodes - done)
        actions = env.sample_actions(n, seed=seed + i)
        episodes = env.step(actions)
        paths.append(save(out / f"shard_{i:05d}.npz", actions, episodes))
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
