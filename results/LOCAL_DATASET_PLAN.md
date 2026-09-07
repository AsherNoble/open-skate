# Balanced local pixel-dataset plan

This earlier large collection plan is retained for reference. The current small,
end-to-end CPU experiment is `./run-local`; see [the research protocol](LOCAL_RESEARCH_PROTOCOL.md).
Current shards use format 4, and the batch collector now supports `--backend
auto|classic|warp`. The historical Warp-only host restriction below no longer
applies to Classic collection.

The training matrix is the full 3×3 product:

| Park | Day | Indoor | Overcast |
|---|---:|---:|---:|
| flat | 4 shards | 4 shards | 4 shards |
| plaza | 4 shards | 4 shards | 4 shards |
| SLS | 4 shards | 4 shards | 4 shards |

At the default 64 episodes per shard this is 36 shards and 2,304 episodes,
exactly 256 episodes per park/appearance domain. The shard is the balancing
unit: train/validation splitting should take the same number of complete shards
from each cell, never split adjacent frames from one episode across sets.

Inspect the deterministic schedule without allocating a renderer:

```bash
.venv/bin/python -m bench.collect_local --dry-run
```

On a local MJX Warp-capable render host, collect it with:

```bash
.venv/bin/python -m bench.collect_local \
  --parks flat plaza sls \
  --appearances day indoor overcast \
  --shards-per-domain 4 \
  --episodes-per-shard 64 \
  --out data/pixel-rollouts
```

No Modal or remote service is involved. Each domain compiles one reusable
`GestureEnv(pixels=True, park=..., appearance=...)`; its four shards receive
distinct deterministic seeds. Files are partitioned as
`park=<park>/appearance=<appearance>/shard_NNNNN.npz`. Every shard embeds both
values in format-v3 metadata, and `manifest.json` records the complete schedule.

The current Apple Silicon development host can compile and CPU-render all nine
MJCF combinations, but the training batch renderer requires the optional MJX
Warp backend, which is not installed here. The plan is therefore validated
locally in dry-run/model-compilation mode; full batch collection should run on a
local supported GPU host. This is an unresolved backend limitation, not a reason
to submit a paid job.
