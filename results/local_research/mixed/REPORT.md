# Local research baseline: mixed

Split episodes: {'train': 104, 'validation': 40, 'test': 40, 'domain': 12}. Horizon 0.2 s; model predicts 8×16 RGB from 64×128 captures.

| Partition | Model MSE | Persistence | Mean frame | No action | Shuffled action |
|---|---:|---:|---:|---:|---:|
| validation | 0.006491 | 0.013426 | 0.044445 | 0.006580 | 0.011828 |
| validation_sim | 0.008257 | 0.009197 | 0.056884 | 0.008880 | 0.008903 |
| validation_device | 0.004730 | 0.017644 | 0.032039 | 0.004286 | 0.004811 |
| test | 0.008303 | 0.013741 | 0.056342 | 0.007986 | 0.013134 |
| test_sim | 0.011626 | 0.011058 | 0.076708 | 0.011937 | 0.011816 |
| test_device | 0.005011 | 0.016398 | 0.036166 | 0.004072 | 0.004896 |
| domain | 0.010388 | 0.009139 | 0.063218 | 0.010076 | 0.010503 |
| domain_sim | 0.010388 | 0.009139 | 0.063218 | 0.010076 | 0.010503 |

Prediction sheets: current, actual +0.2 s, predicted +0.2 s, absolute error. Pixels are enlarged with nearest-neighbour; they are not high-resolution predictions.

Physical MAEs and gesture-search validation are in metrics.json. Model selection uses validation only. Entire scene/capture groups and shared frames stay together.

- Small paired tail-gesture distribution; not full trick coverage.
- Physical outcomes use simulator labels; device outcomes remain unknown.
- No phone execution or measured sim-to-real transfer.
- Jetson Nanos and CUDA throughput have not been benchmarked.
