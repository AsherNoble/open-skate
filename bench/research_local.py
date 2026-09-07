"""One local command: collect/resume, import, split, train, evaluate, report."""
import argparse
from pathlib import Path
import platform
import time
import json
import numpy as np

from opensk.rl import store
from opensk.rl.dataset import collect_dataset,audit
from opensk.rl.backend import warp_status
from opensk.rl.research import experiment


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='data/local-research')
    parser.add_argument('--output',default='results/local_research')
    parser.add_argument('--groups',type=int,default=8)
    parser.add_argument('--episodes',type=int,default=2)
    parser.add_argument('--seconds',type=float,default=1.2)
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--experts',type=Path)
    parser.add_argument('--mode',choices=('synthetic','expert','mixed','all'),default='all')
    parser.add_argument('--skip-search',action='store_true')
    parser.add_argument('--stop-after',type=int,help='test resumability after N newly committed shards')
    args=parser.parse_args(argv)
    root=Path(args.data); output=Path(args.output)
    began=time.perf_counter()
    supported,reason=warp_status()
    print(f'Classic CPU collection; Warp status: {reason}',flush=True)
    collection=collect_dataset(root/'synthetic',groups=args.groups,episodes=args.episodes,
                               seconds=args.seconds,seed=args.seed,stop_after=args.stop_after)
    validation=audit(root/'synthetic')
    roots=[root/'synthetic']
    expert=None
    if args.experts:
        from opensk.rl.experts import import_captures
        expert=import_captures(args.experts,root/'expert')
        roots.append(root/'expert')
    if args.mode in ('mixed','expert') and expert is None:
        parser.error('--experts is required for mixed/expert mode')
    modes=(['synthetic','expert','mixed'] if expert else ['synthetic']) if args.mode=='all' else [args.mode]
    reports={}
    for mode in modes:
        print(f'training {mode}',flush=True)
        reports[mode]=experiment(roots,output/mode,mode=mode,
                                 run_search=not args.skip_search and mode=='synthetic')
    counts={}; visibility={}; hashes={}
    for shard in store.iter_shards(root/'synthetic'):
        domain=f'{shard.park}/{shard.appearance}'
        counts[domain]=counts.get(domain,0)+len(shard)
        visibility.setdefault(domain,[]).extend(shard.extras['obstacle_pixels'].max(axis=1).tolist())
        hashes.setdefault(domain,[]).append(store.array_hash(shard.rgb))
    result=dict(machine=platform.platform(),backend='classic',warp_status=reason,
                collection=collection,audit=validation,episodes_per_domain=counts,
                minimum_max_obstacle_pixels={k:min(v) for k,v in visibility.items()},
                domain_rgb_hashes={k:__import__('hashlib').sha256(''.join(v).encode()).hexdigest() for k,v in hashes.items()},
                experts_imported=expert['imported'] if expert else 0,
                reports={k:str(output/k/'REPORT.md') for k in reports},
                elapsed_s=time.perf_counter()-began,paid_compute=False)
    store.atomic_json(output/'run.json',result)
    lines=['# Local Open Skate experiment','',f"Backend: Classic MuJoCo on {platform.platform()}.",
           f"{sum(counts.values())} synthetic episodes across {len(counts)} domains; {result['experts_imported']} device captures.",
           f"Collection created {collection['new_shards']} shards; existing shards were verified and reused.",'',
           'Each experiment report contains held-out image metrics, physical outcome errors, and sample predictions. A lower image loss alone does not prove action understanding.','']
    for mode,r in reports.items():
        s=r['partitions']['test']
        lines.extend([f"- [{mode}]({mode}/REPORT.md): held-out MSE {s['model_mse']:.6f}; persistence {s['persistence_mse']:.6f}; no-action model {s['no_action_model_mse']:.6f}."])
    lines.extend(['', 'Limits: small action distribution and low-resolution linear forecasts. Device physical outcomes are unlabelled. No policy has been executed on a phone. Jetson Nanos have not been benchmarked; no performance claim is made. CUDA is optional and no paid compute is launched.'])
    with store.atomic_file(output/'REPORT.md') as stream:
        stream.write(('\n'.join(lines)+'\n').encode())
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
