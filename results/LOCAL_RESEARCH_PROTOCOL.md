# Local research protocol

`./run-local` bootstraps only public Python dependencies and performs all computation
on the current computer. The default is Classic MuJoCo, including the same fitted
FollowCamera and TouchModel. CUDA is optional; the older batch collector accepts
`--backend auto|classic|warp`. The Warp entry point refuses macOS before loading
its render kernel. It never retries the observed Apple Silicon native crash.
No remote jobs, Modal calls, or paid services are launched.

`requirements-local.txt` pins the tested numerical/rendering stack for the local
command. It excludes Modal, CUDA and Warp. This avoids silently changing the
research environment when a clean checkout installs newer NumPy or JAX versions.

## Data and recovery

The default experiment writes 72 atomic shards, two gestures per shard, across
nine park/appearance domains and eight scene seeds. Paired gestures share the
same initial scene and appearance draw. Seeds vary lighting, concrete and deck
materials and small starting-position offsets. Two approach anchors per park
change which existing modules occupy the camera view. Obstacle composition is
therefore varied by approach, not by moving collision geometry or drawing
non-collidable fake obstacles. The wide SLS course is approached near its
funbox/manual-pad region instead of its empty reset crop. Flat intentionally
has no obstacles. Non-flat episodes must contain at least nine obstacle pixels
in a segmentation frame; the run report records the measured minimum.

Shards carry RGB, deck-only masks, exact physics-step frame times, physical
outcomes, validity, recipe features, seeds, episode/group IDs and park/appearance.
Array checksums cover dtype, shape and bytes. The manifest pins the collection
configuration and final file checksums. Writes use fsync plus a same-directory
atomic rename. A file lock prevents concurrent experiment collectors. A crash
before rename can leave `.partial` files, which are listed and excluded. A crash
after shard rename but before manifest update is recovered by validating and
adopting the shard. Corrupt, missing committed or mismatched shards fail explicitly;
they are never silently deleted or overwritten. A changed config needs a new root.

Formats 1–3 are read without modifying their files. The migration command writes
version 4 into a separate tree and records original hashes. Pre-spin action
vectors acquire an explicitly disabled spin block. Missing domain information
stays unknown. Legacy files without reliable recording/group/time metadata need
enrichment before training; guessing it could leak capture frames between splits.

## Evaluation

The world model predicts 0.2 seconds ahead at 8×16 RGB using a 16-component
training-only PCA of the current image and ridge regression of the image change.
Features include the exact recipe descriptor and time interactions. Model and
no-action ablation each select ridge strength on validation data only. Persistence
and training-mean-frame scores use precisely the same test pairs. Whole episodes
and their shared scene/capture groups are split before creating frame pairs.
Exact shared frames join groups before splitting. Overcast test scenes are
withheld from training and validation. Model files can be reloaded using
`RidgeWorld.load`; outcome weights are stored separately.

The physical head predicts accumulated roll/yaw, peak height, airtime and maximum
travel from gesture features. Its mean-outcome comparator is fit on the same
training set. Device captures have no trustworthy physical labels and never
supervise this head. Search ranks twelve new gestures by predicted peak height,
then executes all twelve in Classic MuJoCo to measure the selected result, the
random-choice expectation and the oracle best. This does not establish phone
transfer. RGB action shuffling and paired initial-scene effects are reported;
any loss to the no-action ablation remains visible.

Physical trajectories use every simulation substep. Fitted parameters, camera,
touch rays, masses, inertia and authoritative collisions are unchanged. Existing
trajectory hashes and silhouette tests still protect those contracts. Seeded
CPU trajectories reproduce exactly; RGB byte identity is supported on the same
renderer, but another GPU/driver can vary antialiasing. Manifest checksums detect
such changes rather than treating them as equivalent data.

## Hardware and limitations

The measured Mac path uses local OpenGL with Classic MuJoCo. macOS sandboxed
processes may need display access to create a CoreGraphics connection. Renderer
implementation errors fail; tests skip only specifically identified unavailable
GL backends. Optional Warp packages do not make an Apple GPU a CUDA GPU.

Jetson Nanos were not available as configured endpoints in this workspace. They
have not been benchmarked, and no usefulness/throughput claim is made. Run the
same local command on a candidate device and retain its `run.json` before making
any performance comparison. This small linear baseline and narrow gesture prior
are an experiment, not a general world model or a demonstrated sim-to-real policy.
