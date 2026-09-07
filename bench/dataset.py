"""Inspect or migrate older datasets, retaining every original shard."""
import argparse
import json
from pathlib import Path
from opensk.rl import store


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--migrate-to',type=Path)
    args=parser.parse_args()
    if args.migrate_to and (args.migrate_to.resolve()==args.source.resolve()
                           or args.source.resolve() in args.migrate_to.resolve().parents):
        parser.error('migration destination must be outside the original dataset')
    report=[]
    paths=sorted(args.source.rglob('*.npz')) if args.source.is_dir() else [args.source]
    for path in paths:
        shard=store.load(path)
        entry=dict(path=str(path),episodes=len(shard),version=shard.metadata['version'],
                   park=shard.park,appearance=shard.appearance,sha256=store.file_hash(path))
        if args.migrate_to:
            rel=path.relative_to(args.source) if args.source.is_dir() else Path(path.name)
            target=args.migrate_to/rel
            if target.exists():
                got=store.load(target)
                if got.metadata.get('original_sha256')!=entry['sha256']:
                    raise ValueError(f'migration resume mismatch: {target}')
            else:
                store.migrate(path,target)
            entry['migrated_to']=str(target)
        report.append(entry)
    print(json.dumps(dict(shards=report,incomplete_files=[str(p) for p in args.source.rglob('*.partial')]),indent=2))


if __name__=='__main__':
    main()
