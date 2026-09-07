# Local Open Skate experiment

Backend: Classic MuJoCo on macOS-26.2-arm64-arm-64bit.
144 synthetic episodes across 9 domains; 88 device captures.
Collection created 0 shards; existing shards were verified and reused.

Each experiment report contains held-out image metrics, physical outcome errors, and sample predictions. A lower image loss alone does not prove action understanding.

- [synthetic](synthetic/REPORT.md): held-out MSE 0.010891; persistence 0.011058; no-action model 0.010304.
  Synthetic-to-device transfer MSE: 0.031961; persistence 0.016398. No device data entered training or model selection.
  Gesture search: selected 0.190 m actual peak; random-choice expectation 0.089 m; oracle best 0.274 m across 12 new candidates.
- [expert](expert/REPORT.md): held-out MSE 0.001653; persistence 0.016398; no-action model 0.001514.
- [mixed](mixed/REPORT.md): held-out MSE 0.008303; persistence 0.013741; no-action model 0.007986.

synthetic: the no-action ablation is at least as strong as the action-conditioned RGB model on test episodes.

synthetic: held-out overcast does not beat persistence.

expert: the no-action ablation is at least as strong as the action-conditioned RGB model on test episodes.

mixed: the no-action ablation is at least as strong as the action-conditioned RGB model on test episodes.

mixed: held-out overcast does not beat persistence.

Limits: small action distribution and low-resolution linear forecasts. Device physical outcomes are unlabelled. No policy has been executed on a phone. Jetson Nanos have not been benchmarked; no performance claim is made. CUDA is optional and no paid compute is launched.
