"""Versioned, atomic local dataset plans. Completed shards are verified on resume."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np

from . import store
from .action import to_recipe, DURATION_MIN, DURATION_MAX, EASING_MIN, EASING_MAX
from ..sim.gesture_spec import schedule_recipe, spin_window
from ..sim.model.parks import PARKS
from ..sim.appearance import APPEARANCE_PRESETS


def recipe_features(recipe):
    """Fixed physical recipe descriptor, shared with arbitrary device waypoints.

    Three slots, four samples per path, start/duration per slot, spin window.
    No inverse squash of expert actions and no guess at a rig parameter layout.
    """
    schedule = schedule_recipe(recipe)
    if len(schedule) > 3:
        raise ValueError("baseline supports at most three gesture slots")
    out = np.zeros(33, np.float32)
    for i, (start, path) in enumerate(schedule):
        out[i*10:i*10+8] = np.concatenate([path.position_at(t*path.duration)
                                          for t in np.linspace(0, 1, 4)])
        out[i*10+8:i*10+10] = start, path.duration
    end = max((s+p.duration for s,p in schedule), default=0.)
    spin = spin_window(recipe, end)
    if spin:
        out[-3:] = 1., *spin
    return out


def controlled_actions(seed, n=2):
    """Bounded tail gestures with opposite lateral drags per scene.

    A narrow, explicit research distribution; it is not a claim of whole-game
    coverage. The unchanged fitted touch model determines every outcome.
    """
    rng = np.random.default_rng(seed)
    def inv(value, lo, hi):
        return np.arctanh(np.clip(2*(value-lo)/(hi-lo)-1, -.9999, .9999))
    actions = []
    for i in range(n):
        sign = -1 if i % 2 == 0 else 1
        x0 = .5 + rng.uniform(-.015, .015)
        y0 = rng.uniform(.62, .70)
        dx = sign * rng.uniform(.06, .18)
        dy = rng.uniform(-.20, .08)
        pts = [(x0,y0), (x0+dx*.5,y0+dy*.5), (x0+dx,y0+dy)]
        vec = []
        for x,y in pts:
            vec.extend([inv(x,.12,1.), inv(y,.12,.88)])
        vec.extend([inv(rng.uniform(.15,.40),DURATION_MIN,DURATION_MAX),
                    inv(1.,EASING_MIN,EASING_MAX)])
        for _ in range(3):
            vec.extend([inv(.9,.12,1.), inv(.2,.12,.88)])
        vec.extend([inv(.06,DURATION_MIN,DURATION_MAX), inv(1.,EASING_MIN,EASING_MAX)])
        vec.extend([0., -1., 0., 0.])
        actions.append(vec)
    return np.asarray(actions)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def plan_dataset(root, *, groups=8, episodes=2, seconds=1.2, seed=2026):
    if groups < 4 or episodes < 2 or seconds <= 0:
        raise ValueError("need >=4 scene groups, >=2 gestures and positive seconds")
    config = dict(groups=groups, episodes=episodes, seconds=seconds, seed=seed,
                  parks=list(PARKS), appearances=list(APPEARANCE_PRESETS),
                  collector_version=1, backend="classic")
    digest = fingerprint(config)
    items = []
    for park in PARKS:
        for appearance in APPEARANCE_PRESETS:
            for group in range(groups):
                items.append(dict(path=f"park={park}/appearance={appearance}/shard_{group:05d}.npz",
                                  park=park, appearance=appearance, group=group,
                                  seed=seed+group, episodes=episodes))
    return dict(version=1, config=config, config_sha256=digest, items=items, completed={})


def validate_manifest(manifest):
    if manifest.get("version") != 1:
        raise ValueError("unsupported manifest version")
    config = manifest["config"]
    expected = plan_dataset('.', **{k: config[k] for k in ('groups','episodes','seconds','seed')})
    if (manifest['config_sha256'] != fingerprint(config) or config != expected['config']
            or manifest['items'] != expected['items']):
        raise ValueError("manifest plan/config mismatch")
    paths = {item['path'] for item in expected['items']}
    if not set(manifest['completed']).issubset(paths):
        raise ValueError("unknown completed shard in manifest")


def collect_dataset(root, *, groups=8, episodes=2, seconds=1.2, seed=2026,
                    stop_after=None):
    from .classic import ClassicEnv
    root = Path(root)
    requested = plan_dataset(root, groups=groups, episodes=episodes, seconds=seconds, seed=seed)
    path = root / 'manifest.json'
    # Prevent concurrent writers; flock releases automatically after SIGKILL.
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.collection.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = json.loads(path.read_text()) if path.exists() else requested
        validate_manifest(manifest)
        if manifest['config'] != requested['config']:
            raise ValueError("resume config differs; use a separate output directory")
        store.atomic_json(path, manifest)
        new = 0
        started = time.perf_counter()
        for item in manifest['items']:
            target = root / item['path']
            identity = dict(item, config_sha256=manifest['config_sha256'])
            if target.exists():
                shard = store.load(target)
                if (shard.metadata.get('collection') != identity or len(shard) != episodes
                        or shard.park != item['park'] or shard.appearance != item['appearance']):
                    raise ValueError(f"resume metadata mismatch: {target}")
                digest = store.file_hash(target)
                old = manifest['completed'].get(item['path'])
                if old is not None and old != digest:
                    raise ValueError(f"manifest checksum mismatch: {target}")
            else:
                if item['path'] in manifest['completed']:
                    raise ValueError(f"completed shard is missing: {target}")
                env = ClassicEnv(park=item['park'], appearance=item['appearance'],
                                 seconds=seconds, vary=True)
                actions = controlled_actions(item['seed'], episodes)
                recipes = [to_recipe(a) for a in actions]
                pairs = [env.run_recipe(recipe, item['seed']) for recipe in recipes]
                from .env import Episodes
                ep = Episodes(*(np.concatenate([getattr(r,k) for r,_ in pairs]) for k in Episodes._fields))
                extra = {k: np.stack([e[k] for _,e in pairs]) for k in pairs[0][1]}
                extra['action_features'] = np.stack([recipe_features(r) for r in recipes])
                extra['outcome_known'] = np.ones(episodes, bool)
                # Group covers ALL appearances and paired gestures of a scene.
                extra['group_id'] = np.array([f"sim:{seed}:{item['group']}"]*episodes)
                extra['episode_id'] = np.array([fingerprint([identity, i]) for i in range(episodes)])
                if item['park'] != 'flat' and not np.all(extra['obstacle_pixels'].max(axis=1) > 8):
                    raise ValueError(f"park not visible: {item['park']} anchor seed {item['seed']}")
                store.save(target, actions, ep, park=item['park'], appearance=item['appearance'],
                           metadata=dict(collection=identity, recipes=recipes, backend='classic',
                                         obstacle_composition='existing park, two approach anchors',
                                         action_distribution='paired-tail-v1'), extras=extra)
                digest = store.file_hash(target)
                new += 1
                print(f"collected {item['path']}: {int(ep.valid.sum())}/{episodes} valid", flush=True)
            manifest['completed'][item['path']] = digest
            store.atomic_json(path, manifest)
            if stop_after is not None and new >= stop_after:
                raise InterruptedError("requested interruption after committed shard")
        return dict(new_shards=new, total_shards=len(manifest['items']),
                    elapsed_s=time.perf_counter()-started,
                    incomplete_files=[str(p.relative_to(root)) for p in root.rglob('*.partial')])


def audit(root):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    validate_manifest(manifest)
    for item in manifest['items']:
        path = root / item['path']
        if not path.exists():
            raise ValueError(f"incomplete dataset: {path}")
        shard = store.load(path)
        if shard.metadata.get('collection') != dict(item,config_sha256=manifest['config_sha256']):
            raise ValueError(f"metadata mismatch: {path}")
        if manifest['completed'].get(item['path']) != store.file_hash(path):
            raise ValueError(f"manifest checksum mismatch: {path}")
    actual = {str(p.relative_to(root)) for p in root.rglob('*.npz')}
    if actual != {i['path'] for i in manifest['items']}:
        raise ValueError("unmanifested shards in dataset")
    return dict(shards=len(actual), incomplete_files=[str(p) for p in root.rglob('*.partial')])
