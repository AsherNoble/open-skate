# Local research baseline: expert

Split episodes: {'train': 56, 'validation': 16, 'test': 16, 'domain': 0}. Horizon 0.2 s; model predicts 8×16 RGB from 64×128 captures.

| Partition | Model MSE | Persistence | Mean frame | No action | Shuffled action |
|---|---:|---:|---:|---:|---:|
| validation | 0.001610 | 0.017644 | 0.021087 | 0.001620 | 0.001659 |
| validation_device | 0.001610 | 0.017644 | 0.021087 | 0.001620 | 0.001659 |
| test | 0.001653 | 0.016398 | 0.025222 | 0.001514 | 0.001672 |
| test_device | 0.001653 | 0.016398 | 0.025222 | 0.001514 | 0.001672 |

Prediction sheets: current, actual +0.2 s, predicted +0.2 s, absolute error. Pixels are enlarged with nearest-neighbour; they are not high-resolution predictions.

Physical MAEs and gesture-search validation are in metrics.json. Model selection uses validation only. Entire scene/capture groups and shared frames stay together.

- Small paired tail-gesture distribution; not full trick coverage.
- Physical outcomes use simulator labels; device outcomes remain unknown.
- No phone execution or measured sim-to-real transfer.
- Jetson Nanos and CUDA throughput have not been benchmarked.
