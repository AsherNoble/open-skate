import json
from pathlib import Path
import numpy as np
import cv2
import pytest

from opensk.rl.experts import capture_recipe,import_captures,read_episodes,split_episodes
from opensk.rl.dataset import recipe_features
from opensk.rl.research import pairs,RidgeWorld,pixel_scores,experiment
from opensk.rl import store


def test_device_vector_and_absolute_spin_preserved():
    meta=dict(params=[.4,.7,.5,.6,.6,.5,.2,1., 1.,.8,.2])
    recipe=capture_recipe(meta)
    assert recipe['gestures'][0]['points'][0]==[.4,.7]
    assert recipe['spin']==dict(enabled=True,t_start=.2,t_end=.8)
    assert recipe_features(recipe).shape==(33,)
    absolute=capture_recipe(dict(waypoints=[[.4,.7],[.6,.5]],duration=.2,
                                 spin_active=True,spin_hold_start_s=.1,spin_hold_end_s=.4))
    np.testing.assert_allclose(recipe_features(absolute)[-3:],[1,.1,.4])


def test_import_resume_unknown_labels_and_provenance(tmp_path):
    source=tmp_path/'captures'; folder=source/'session'/'episode'; folder.mkdir(parents=True)
    meta=dict(waypoints=[[.4,.7],[.6,.5]],duration=.2,frame_times=[0.,.1,.2],
              park='reported park',session='recording-a',capture_offset_source='fallback')
    (folder/'meta.json').write_text(json.dumps(meta))
    for i in range(3):
        cv2.imwrite(str(folder/f'frame_{i:03d}.png'),np.full((128,64,3),40+i,np.uint8))
    out=tmp_path/'import'
    r=import_captures(source,out,gameplay_filter=lambda image:True)
    path=out/r['records'][0]['path']; sha=store.file_hash(path)
    import_captures(source,out,gameplay_filter=lambda image:True)
    assert store.file_hash(path)==sha
    s=store.load(path)
    assert not s.extras['outcome_known'].any()
    assert s.park=='unknown' and s.metadata['provenance']['park_uncertain']
    assert s.metadata['provenance']['original_metadata']==meta
    assert len(read_episodes([out,out]))==1


def records():
    result=[]
    for group in range(8):
        for appearance in ('day','indoor','overcast'):
            for action in (-1,1):
                rng=np.random.default_rng(group)
                base=rng.uniform(.15,.25,(128,64,3))
                frames=np.stack([base+action*.09*t for t in np.arange(16)/15])
                rgb=np.uint8(frames*255)
                # Unique domain/scene content prevents accidental frame alias.
                rgb[:,0,0,0]=group*3+('day','indoor','overcast').index(appearance)
                result.append(dict(id=f'{group}/{appearance}/{action}',group=f'g{group}',source='sim',
                                   appearance=appearance,park='flat',features=np.array([action]+[0]*32,float),
                                   rgb=rgb,times=np.arange(16)/30,frame_valid=np.ones(16,bool),
                                   known=True,outcomes=np.array([0,0,action*.1,0,0]),
                                   hashes=[store.array_hash(f) for f in rgb]))
    return result


def test_splits_keep_scenes_and_duplicate_frames_together():
    rec=records()
    rec[0]['hashes'].append('shared-frame')
    rec[6]['hashes'].append('shared-frame')
    split=split_episodes(rec,mode='synthetic')
    sets={k:{r['group'] for r in v} for k,v in split.items()}
    assert not sets['train'] & (sets['validation']|sets['test']|sets['domain'])
    assert not sets['validation'] & (sets['test']|sets['domain'])
    assert all(r['appearance']!='overcast' for r in split['train'])
    owners={}
    for part in ('train','validation','test'):
        for r in split[part]:
            for digest in r['hashes']:
                assert digest not in owners or owners[digest]==part
                owners[digest]=part


def test_learning_recovers_controlled_action_effect(tmp_path):
    split=split_episodes(records(),mode='synthetic')
    train=pairs(split['train']); held=pairs(split['test'])
    model=RidgeWorld(train).fit(.1)
    ablated=RidgeWorld(train,use_action=False).fit(.1)
    scores=pixel_scores(model,held,ablated)
    assert scores['model_mse']<scores['persistence_mse']*.2
    assert scores['shuffled_action_mse']>scores['model_mse']*2
    assert scores['action_prediction_change_mse']>1e-6
    model.save(tmp_path/'model.npz')
    np.testing.assert_array_equal(model.predict(held),RidgeWorld.load(tmp_path/'model.npz').predict(held))


def test_actual_end_to_end_resume_train_evaluate(tmp_path):
    from opensk.pose.render import SceneRenderer,gl_backend_unavailable
    from opensk.sim.core import SkateSim
    try:
        renderer=SceneRenderer(SkateSim(),height=8,width=8)
    except Exception as exc:
        if gl_backend_unavailable(exc):
            pytest.skip(f'GL unavailable: {exc}')
        raise
    renderer._renderer.close()
    from bench.research_local import main
    args=['--data',str(tmp_path/'data'),'--output',str(tmp_path/'output'),
          '--groups','4','--seconds','.3','--skip-search']
    with pytest.raises(InterruptedError):
        main(args+['--stop-after','1'])
    main(args)
    report=json.loads((tmp_path/'output'/'synthetic'/'metrics.json').read_text())
    assert report['partitions']['test']['pairs']>0
    first={str(p):store.file_hash(p) for p in (tmp_path/'data').rglob('*.npz')}
    main(args)
    assert first=={str(p):store.file_hash(p) for p in (tmp_path/'data').rglob('*.npz')}
    run=json.loads((tmp_path/'output'/'run.json').read_text())
    assert run['collection']['new_shards']==0
    assert len(set(run['domain_rgb_hashes'].values()))==9
    assert (tmp_path/'output'/'synthetic'/'test_predictions.png').exists()
