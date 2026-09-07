"""Measure the actual Classic collector on this machine; no remote execution."""
import argparse
import platform
import time
from pathlib import Path
import numpy as np

from opensk.rl.classic import ClassicEnv
from opensk.rl.dataset import controlled_actions
from opensk.rl.action import to_recipe
from opensk.rl import store


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='results/local_research/benchmark.json')
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args(argv)
    if args.repeats<2:
        parser.error('need >=2 repeats to measure reproducibility')
    times=[]; baseline=None
    for i in range(args.repeats):
        start=time.perf_counter()
        env=ClassicEnv(park='plaza',appearance='day',seconds=1.2,vary=True)
        pairs=[env.run_recipe(to_recipe(a),2026) for a in controlled_actions(2026)]
        times.append(time.perf_counter()-start)
        signature=[{k:store.array_hash(v) for k,v in ep._asdict().items()} for ep,_ in pairs]
        if baseline is not None and signature!=baseline:
            raise RuntimeError('same-device seeded collection failed exact reproduction')
        baseline=signature
    # Compare with an independently collected on-disk shard when available.
    dataset=Path('data/local-research/synthetic/park=plaza/appearance=day/shard_00000.npz')
    disk_match=None
    if dataset.exists():
        old=store.load(dataset)
        disk_match=all(np.array_equal(getattr(old,k),np.concatenate([getattr(ep,k) for ep,_ in pairs]).astype(getattr(old,k).dtype))
                       for k in ('pos','quat','rgb','roll_deg','yaw_deg','peak_height','air_s','displacement','valid'))
        if not disk_match:
            raise RuntimeError('seeded collection differs from independently collected shard')
    report=dict(machine=platform.platform(),backend='classic',episodes_per_trial=2,
                episode_seconds=1.2,frames_per_episode=37,seconds_per_trial=times,
                median_episodes_per_second=2/float(np.median(times)),
                median_frames_per_second=74/float(np.median(times)),
                exact_repeated_trajectory_and_rgb=True,independent_saved_shard_matches=disk_match,
                includes='scene construction, settling, touch physics, RGB, deck masks and obstacle segmentation',
                jetson='not measured; run this command on each Nano before claiming utility')
    store.atomic_json(args.output,report)
    print(report)


if __name__=='__main__':
    main()
