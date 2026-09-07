# Local research baseline: synthetic

Split episodes: {'train': 48, 'validation': 24, 'test': 24, 'domain': 12, 'device_transfer': 16}. Horizon 0.2 s; model predicts 8×16 RGB from 64×128 captures.

| Partition | Model MSE | Persistence | Mean frame | No action | Shuffled action |
|---|---:|---:|---:|---:|---:|
| validation | 0.008059 | 0.009197 | 0.029816 | 0.007142 | 0.009164 |
| validation_sim | 0.008059 | 0.009197 | 0.029816 | 0.007142 | 0.009164 |
| test | 0.010891 | 0.011058 | 0.053766 | 0.010304 | 0.011720 |
| test_sim | 0.010891 | 0.011058 | 0.053766 | 0.010304 | 0.011720 |
| domain | 0.009292 | 0.009139 | 0.053450 | 0.008836 | 0.009891 |
| domain_sim | 0.009292 | 0.009139 | 0.053450 | 0.008836 | 0.009891 |
| device_transfer | 0.031961 | 0.016398 | 0.105900 | 0.019603 | 0.032026 |
| device_transfer_device | 0.031961 | 0.016398 | 0.105900 | 0.019603 | 0.032026 |

Prediction sheets: current, actual +0.2 s, predicted +0.2 s, absolute error. Pixels are enlarged with nearest-neighbour; they are not high-resolution predictions.

Physical MAEs and gesture-search validation are in metrics.json. Model selection uses validation only. Entire scene/capture groups and shared frames stay together.

- Small paired tail-gesture distribution; not full trick coverage.
- Physical outcomes use simulator labels; device outcomes remain unknown.
- No phone execution or measured sim-to-real transfer.
- Jetson Nanos and CUDA throughput have not been benchmarked.
