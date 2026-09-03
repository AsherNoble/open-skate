# Does a model trained on this environment predict this environment?

The plan's verification step. Not a research result — a pipeline check. If a
model trained on what Open Skate produces cannot predict what Open Skate
produces, something upstream is broken and no architecture will rescue it.

## The setup

Two independent 64-episode batches on an A10G, different seeds. Train on one,
score on the other. **The split is across episodes, never across frames**:
consecutive frames within an episode are nearly identical, so a frame-level
split leaks the answer and every model looks excellent.

Linear least squares on `[frame, action, 1]` → next frame. Linear on purpose:
it has a closed form so there is no training loop to get wrong, and if a linear
map on the action already beats persistence then the pipeline carries real
action-conditioned signal. A deep model beating persistence would leave that
ambiguous.

Frames are downsampled 8× to 16×8×3 = 384 features. Not cosmetic: at full
resolution it is 24,576 features against ~3,500 rows, and a fit that fat
interpolates the training set exactly — it would "beat" persistence by
memorising, which is the precise failure this check exists to detect.
`evaluate` now refuses an underdetermined fit rather than returning a number.

## The result

| | MSE per pixel |
|---|---|
| linear model | **0.00249** |
| persistence (next frame = this frame) | 0.00252 |
| mean training frame | 0.00644 |

**It beats persistence, by 1.2%.** Read that as a floor, not a headline:

- **The pipeline works.** Frames, actions and outcomes come out the far end
  aligned and carrying signal, and the model is 2.6× better than the
  mean-frame baseline, so it is far from trivial.
- **The margin is small.** Persistence is a strong baseline at 30 fps — most of
  a frame genuinely does not change in 33 ms — and 1.2% is close enough to
  noise that no stronger claim is warranted from one pair of batches.
- **A linear map on 384 pixels is a crude probe.** The interesting question,
  which this does not answer, is how much of the remaining error a model with
  actual capacity can take out.

## What would make this a real result

More batches (the margin needs an error bar), a nonlinear model, and a longer
prediction horizon than one frame — a world model that only predicts 33 ms
ahead is not doing the job the project needs.
