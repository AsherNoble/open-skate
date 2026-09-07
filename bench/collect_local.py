"""Plan or collect balanced local pixel shards without Modal.

The default matrix is the full Cartesian product of the three authoritative
parks and three neutral appearances. Use ``--dry-run`` to inspect the plan.
Automatic backend selection uses Classic CPU rendering when CUDA is unavailable.
For the complete experiment with varied approaches and learning, use run-local.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from opensk.sim.appearance import APPEARANCE_PRESETS
from opensk.sim.model.parks import PARKS


def make_plan(*, out: str | Path, parks: list[str], appearances: list[str],
              shards_per_domain: int, episodes_per_shard: int,
              seed: int) -> list[dict]:
    """A deterministic, exactly balanced shard schedule."""
    root = Path(out)
    plan = []
    for domain, (park, appearance) in enumerate(
            (p, a) for p in parks for a in appearances):
        for shard in range(shards_per_domain):
            plan.append({
                "park": park,
                "appearance": appearance,
                "shard": shard,
                "episodes": episodes_per_shard,
                "seed": seed + domain * shards_per_domain + shard,
                "path": str(root / f"park={park}" / f"appearance={appearance}"
                            / f"shard_{shard:05d}.npz"),
            })
    return plan


def collect_plan(plan: list[dict], *, backend="auto") -> None:
    """Execute a plan locally, compiling one reusable environment per domain."""
    import numpy as np

    from opensk.rl import store
    from opensk.rl.env import GestureEnv
    from opensk.rl.backend import select_backend
    from opensk.rl.classic import ClassicEnv
    backend = select_backend(backend)

    active = None
    env = None
    for item in plan:
        target = Path(item['path'])
        if target.exists():
            old = store.load(target)
            if (old.metadata.get('collection') != item or len(old) != item['episodes']
                    or old.park != item['park'] or old.appearance != item['appearance']):
                raise ValueError(f'resume mismatch: {target}')
            continue
        domain = (item["park"], item["appearance"], item["episodes"])
        if domain != active:
            kwargs=dict(batch=item['episodes'],park=item['park'],appearance=item['appearance'])
            env = (ClassicEnv(**kwargs,seconds=2.3) if backend=='classic'
                   else GestureEnv(pixels=True,**kwargs))
            active = domain
        actions = env.sample_actions(item["episodes"], seed=item["seed"])
        episodes = env.step(actions)
        path = store.save(item["path"], actions, episodes,
                          park=env.park_name, appearance=env.appearance,
                          metadata=dict(collection=item,backend=backend))
        kept = int(np.asarray(episodes.valid).sum())
        print(f"{path}: {item['episodes']} episodes, {kept} physical",
              flush=True)


def _choices(values: list[str], available: dict, label: str) -> list[str]:
    unknown = [value for value in values if value not in available]
    if unknown:
        raise ValueError(f"unknown {label}: {', '.join(unknown)}")
    return values


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/pixel-rollouts")
    parser.add_argument("--parks", nargs="+", default=list(PARKS),
                        help=f"choices: {', '.join(PARKS)}")
    parser.add_argument("--appearances", nargs="+",
                        default=list(APPEARANCE_PRESETS),
                        help=f"choices: {', '.join(APPEARANCE_PRESETS)}")
    parser.add_argument("--shards-per-domain", type=int, default=4)
    parser.add_argument("--episodes-per-shard", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backend", choices=("auto","classic","warp"),default="auto")
    args = parser.parse_args(argv)

    parks = _choices(args.parks, PARKS, "parks")
    appearances = _choices(args.appearances, APPEARANCE_PRESETS,
                           "appearances")
    if args.shards_per_domain < 1 or args.episodes_per_shard < 1:
        parser.error("shard and episode counts must be positive")
    plan = make_plan(out=args.out, parks=parks, appearances=appearances,
                     shards_per_domain=args.shards_per_domain,
                     episodes_per_shard=args.episodes_per_shard,
                     seed=args.seed)
    manifest = Path(args.out) / "manifest.json"
    payload = {
        "format": 1,
        "balanced_by": ["park", "appearance"],
        "domains": len(parks) * len(appearances),
        "total_shards": len(plan),
        "total_episodes": sum(item["episodes"] for item in plan),
        "plan": plan,
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return
    manifest.parent.mkdir(parents=True, exist_ok=True)
    from opensk.rl import store
    if manifest.exists() and json.loads(manifest.read_text()) != payload:
        raise ValueError('resume manifest differs; use a new output directory')
    store.atomic_json(manifest,payload)
    collect_plan(plan,backend=args.backend)
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
