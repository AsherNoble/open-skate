"""Import user-owned device captures without inventing simulator ground truth."""
from pathlib import Path
import json
import numpy as np
import cv2
from PIL import Image

from . import store
from .dataset import fingerprint, recipe_features
from .env import Episodes
from ..pose.frames import load_sample, _gameplay_filter


def capture_recipe(meta):
    """Device vectors are already bounded; never squash them as policy logits."""
    if 'recipe' in meta:
        return meta['recipe']
    if meta.get('params'):
        a = np.asarray(meta['params'], float)
        spin = len(a) % 9 == 2
        n = (len(a) - (3 if spin else 0) + 1) // 9
        if n < 1 or len(a) != 9*n-1+(3 if spin else 0):
            raise ValueError('unrecognised device action layout')
        recipe = dict(gestures=[dict(points=a[i*8:i*8+6].reshape(3,2).tolist(),
                                    duration=float(a[i*8+6]), easing_power=float(a[i*8+7]))
                                for i in range(n)], delays=a[8*n:9*n-1].tolist())
        if spin:
            start,end = sorted(np.clip(a[-2:],0,1))
            recipe['spin'] = dict(enabled=bool(a[-3]>=0),t_start=float(start),t_end=float(end))
        return recipe
    recipe = dict(gestures=[dict(points=meta['waypoints'],duration=meta['duration'],
                                easing_power=meta.get('easing_power',1.))],delays=[])
    if meta.get('spin_active'):
        if meta.get('spin_hold_start_s') is None or meta.get('spin_hold_end_s') is None:
            raise ValueError('spin active but hold times unknown')
        recipe['spin'] = dict(enabled=True,spin_hold_start_s=meta['spin_hold_start_s'],
                              spin_hold_end_s=meta['spin_hold_end_s'])
    return recipe


def import_captures(source, destination, *, limit=None, gameplay_filter=None):
    source, destination = Path(source), Path(destination)
    if not source.is_dir():
        raise ValueError(f'capture root missing: {source}')
    gf = gameplay_filter or _gameplay_filter().is_gameplay_frame
    records, rejected = [], []
    for meta_path in sorted(source.rglob('meta.json')):
        if limit is not None and len(records) >= limit:
            break
        meta = json.loads(meta_path.read_text())
        sample = load_sample(meta_path.parent)
        if sample is None:
            rejected.append(dict(path=str(meta_path),reason='missing gesture/frames/timestamps'))
            continue
        try:
            recipe = capture_recipe(meta)
            features = recipe_features(recipe)
        except (ValueError, KeyError) as exc:
            rejected.append(dict(path=str(meta_path),reason=str(exc)))
            continue
        times = sample.frame_times
        if not np.isfinite(times).all() or np.any(np.diff(times)<=0):
            raise ValueError(f'invalid capture timestamps: {meta_path}')
        frames, hashes, keep = [], [], []
        for i,t in enumerate(times):
            frame = sample.frame(i)
            if frame is None:
                raise ValueError(f'undecodable frame {i}: {meta_path}')
            rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            hashes.append(store.array_hash(rgb))
            keep.append(bool(gf(Image.fromarray(rgb))))
            frames.append(cv2.resize(rgb,(64,128),interpolation=cv2.INTER_AREA))
        if sum(keep)<3:
            rejected.append(dict(path=str(meta_path),reason='fewer than 3 gameplay frames'))
            continue
        episode_id = fingerprint([meta,hashes])
        # Conservative grouping: every segment/window from one capture session
        # stays together, including windows with overlapping source timestamps.
        session = meta.get('session') or str(meta_path.parent.parent.relative_to(source))
        provenance = dict(original_metadata=meta,original_path=str(meta_path.resolve()),
                          metadata_sha256=store.file_hash(meta_path),frame_hashes=hashes,
                          recipe=recipe,park_uncertain=True,deck_uncertain=True,
                          park_reported=meta.get('park'),deck_reported=meta.get('deck'),
                          timing_uncertainty=meta.get('capture_offset_source','unknown'))
        f=len(times)
        ep=Episodes(np.zeros((1,f,3)),np.zeros((1,f,4)),
                    *[np.zeros(1) for _ in range(5)],np.ones(1,bool),np.array([frames],np.uint8))
        extra=dict(frame_times=times[None],frame_valid=np.array([keep]),
                   action_features=features[None],outcome_known=np.zeros(1,bool),
                   episode_id=np.array([episode_id]),group_id=np.array([f'device:{session}']),
                   frame_hashes=np.array([hashes]))
        path=destination/f'{episode_id}.npz'
        if path.exists():
            old=store.load(path)
            if old.metadata.get('provenance')!=provenance:
                raise ValueError(f'expert resume provenance mismatch: {path}')
        else:
            # actions uses the common recipe descriptor for expert records;
            # explicit encoding prevents mistaking it for executable logits.
            store.save(path,features[None],ep,source='device',park='unknown',appearance='unknown',
                       metadata=dict(provenance=provenance,actions_encoding='recipe-features-v1'),extras=extra)
        records.append(dict(path=path.name,sha256=store.file_hash(path),episode_id=episode_id))
    report=dict(source=str(source.resolve()),imported=len(records),records=records,rejected=rejected,
                physical_labels='unavailable; excluded from outcome supervision')
    store.atomic_json(destination/'import.json',report)
    return report


def read_episodes(roots):
    records=[]
    ids=set()
    for root in roots:
        for shard in store.iter_shards(root):
            extra=shard.extras or {}
            if 'action_features' not in extra or 'episode_id' not in extra or 'group_id' not in extra:
                raise ValueError('training needs explicit episode identity, grouping and recipe features; migrate/enrich legacy provenance first')
            for i in range(len(shard)):
                eid=str(extra['episode_id'][i])
                if eid in ids:
                    continue
                ids.add(eid)
                if not shard.valid[i]:
                    continue
                rgb=shard.rgb[i]
                records.append(dict(id=eid,group=str(extra['group_id'][i]),source=shard.source,
                                    park=shard.park,appearance=shard.appearance,rgb=rgb,
                                    times=extra['frame_times'][i],features=extra['action_features'][i],
                                    frame_valid=extra.get('frame_valid',np.ones(shard.pos.shape[:2],bool))[i],
                                    known=bool(extra.get('outcome_known',np.zeros(len(shard),bool))[i]),
                                    outcomes=np.array([getattr(shard,k)[i] for k in
                                                      ('roll_deg','yaw_deg','peak_height','air_s','displacement')]),
                                    hashes=list(extra['frame_hashes'][i]) if 'frame_hashes' in extra else
                                           [store.array_hash(f) for f in rgb]))
    return records


def split_episodes(records, *, mode='mixed', holdout='overcast', seed=19):
    """Union sessions, episode IDs and exact shared frames before splitting."""
    if mode not in ('synthetic','expert','mixed'):
        raise ValueError('invalid experiment mode')
    selected=[r for r in records if mode=='mixed' or r['source']==('sim' if mode=='synthetic' else 'device')]
    parent={r['group']:r['group'] for r in selected}
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]
            x=parent[x]
        return x
    owners={}
    for r in selected:
        for key in [r['id'],*r['hashes']]:
            if key in owners:
                a,b=find(r['group']),find(owners[key])
                parent[max(a,b)]=min(a,b)
            else:
                owners[key]=r['group']
    result=dict(train=[],validation=[],test=[],domain=[])
    for source in ('sim','device'):
        groups=sorted({find(r['group']) for r in selected if r['source']==source},
                      key=lambda g:fingerprint([seed,g]))
        if not groups:
            continue
        if len(groups)<3:
            raise ValueError(f'{source}: need >=3 independent capture/scene groups for train/validation/test')
        n=max(1,len(groups)//4)
        val=set(groups[:n]); test=set(groups[n:2*n])
        for r in selected:
            if r['source']!=source:
                continue
            group=find(r['group'])
            partition='validation' if group in val else 'test' if group in test else 'train'
            if source=='sim' and r['appearance']==holdout:
                if partition=='test':
                    result['domain'].append(r)
            else:
                result[partition].append(r)
    return result
