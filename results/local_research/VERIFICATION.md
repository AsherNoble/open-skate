# Verification and measured limits

The local experiment is implemented and exercised through collection, recovery,
capture import, training, evaluation and simulator-validated gesture search.
This establishes an executable research baseline, not successful phone transfer.

## Reproduction

- Final full suite: **190 passed, 0 skipped in 211.41 seconds**. This includes
  renderer-backed regressions, the established trajectory/collision/touch/mask
  guards, atomic recovery, migration, capture import, grouped splits, learning,
  model reload, direct transfer isolation and real local end-to-end resume tests.

- `./run-local` from a fresh local clone with no virtual environment installed
  dependencies, collected all 72 shards / 144 episodes, trained the model,
  evaluated it and validated gesture search. Collection took 66.165 seconds;
  computation after dependency installation took 78.665 seconds.
- The fresh clone produced the same RGB content hashes for all nine domains as
  the original dataset. After pinning the tested numerical stack, a rerun reused
  all 72 shards, created zero new shards, and completed in 9.330 seconds.
- `./run-local --experts data/tricks` additionally imported 88 existing captures
  and ran synthetic-only, expert-only and mixed experiments. Device physical
  outcomes remain explicitly unknown.
- `bench.benchmark_classic` measured **1.902 episodes/s**, including scene
  construction, settling, touch physics, RGB, deck masks and obstacle segmentation.
  Three repeat trials were bit-identical in trajectories and RGB and matched an
  independently collected shard. See `benchmark.json` for all timings.
- Every domain contains 16 episodes. Every plaza episode shows at least 1,534
  obstacle pixels in one training frame; every SLS episode shows at least 2,401.
  Flat intentionally has no obstacles.

## Learning results

| Training source | Held-out RGB MSE | Persistence MSE | No-action model MSE |
|---|---:|---:|---:|
| Synthetic | 0.010891 | 0.011058 | 0.010304 |
| Expert | 0.001653 | 0.016398 | 0.001514 |
| Mixed | 0.008303 | 0.013741 | 0.007986 |

All forecasts are 0.2 seconds ahead at 8×16 RGB, trained from 64×128 observations.
The no-action model wins against the action-conditioned model in every row.
The synthetic model also loses to persistence on held-out overcast:
0.009292 versus 0.009139. Shuffling actions worsens the action model, but this
does not overcome its weakness against the independently trained ablation.

For paired gestures sharing the exact initial frame, the synthetic model's
predicted/actual change has cosine similarity 0.332, but its effect MSE is
0.02577 versus 0.01940 for predicting zero effect. It responds to the gesture;
the response is poorly calibrated. Prediction sheets visibly blur rotation and
obstacle boundaries. The direct synthetic-to-device result is recorded separately
as `device_transfer` in `synthetic/metrics.json`; no device data enters that model's
training, normalisation, PCA or hyperparameter selection.

Direct synthetic-to-device prediction on 16 held-out captures has MSE **0.03196**
versus **0.01640** for persistence and **0.01960** for the no-action model. It fails
the transfer baseline. The dataset/import/evaluation connection is working;
successful visual transfer has not been demonstrated.

The learned physical head selected a new gesture reaching **0.190 m** actual peak
height, versus **0.089 m** for random choice and **0.274 m** for the best of twelve
new candidates. All twelve were evaluated in Classic MuJoCo after ranking. This
is one small search at one seed, not evidence of a robust trick policy. The
selected recipe is preserved in `synthetic/metrics.json`.

## Remaining limitations

- No learned policy has been executed on a phone; physical transfer is unproven.
- The gesture distribution is a small paired tail-drag experiment, and the world
  model is a low-resolution linear baseline. It is not a full-action-space model.
- Appearance varies materials/light and approach anchors. Collision modules are
  not rearranged; approach selects the visible obstacle composition.
- Legacy arrays remain readable and migrate without precision loss. Records
  lacking trustworthy time/group/recipe provenance need enrichment before training.
- Jetson Nanos and CUDA collection throughput were not measured. No claim about
  Nano utility is made. macOS Warp rendering fails early with a clear explanation;
  Classic MuJoCo remains the local path.
- No Modal or paid compute was launched. The fresh verification clone is retained
  in `/private/tmp/open-skate-clean.o3TjiV` for inspection; its data is disposable.
