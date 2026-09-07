import json
import platform
from pathlib import Path
import numpy as np
import pytest

from opensk.rl import store
from opensk.rl.env import Episodes
from opensk.rl.dataset import collect_dataset,audit,plan_dataset,validate_manifest


def ep(b=2,f=3):
    return Episodes(np.zeros((b,f,3)),np.zeros((b,f,4)),
                    *[np.zeros(b) for _ in range(5)],np.ones(b,bool),np.zeros((b,f,128,64,3),np.uint8))


def test_legacy_migration_retains_source_and_disabled_spin(tmp_path):
    p=tmp_path/'legacy.npz'
    data={k:v for k,v in ep()._asdict().items()}
    data['actions']=np.zeros((2,17))
    data['meta']=np.frombuffer(json.dumps(dict(version=1,source='sim')).encode(),np.uint8)
    np.savez_compressed(p,**data)
    digest=store.file_hash(p)
    dest=store.migrate(p,tmp_path/'nested'/'new.npz')
    got=store.load(dest)
    assert got.actions.shape==(2,20) and np.all(got.actions[:,-3]==-1)
    assert got.park==got.appearance=='unknown'
    assert store.file_hash(p)==digest
    assert len(list(store.iter_shards(tmp_path)))==2
    with pytest.raises(ValueError): store.migrate(p,p)


def test_atomic_interruption_keeps_previous_file(tmp_path):
    p=tmp_path/'record.json'
    store.atomic_json(p,dict(old=True))
    with pytest.raises(InterruptedError):
        with store.atomic_file(p) as stream:
            stream.write(b'broken')
            raise InterruptedError()
    assert json.loads(p.read_text())==dict(old=True)
    assert not list(tmp_path.glob('*.partial'))


def test_corruption_and_dimensions_fail(tmp_path):
    p=store.save(tmp_path/'s.npz',np.zeros((2,20)),ep(),park='flat',appearance='day')
    with np.load(p) as z: data={k:z[k] for k in z.files}
    data['actions'][0,0]=10
    np.savez_compressed(p,**data)
    with pytest.raises(ValueError,match='checksum'): store.load(p)
    with pytest.raises(ValueError,match='dimensions'):
        store.save(tmp_path/'bad.npz',np.zeros((3,20)),ep(),park='flat',appearance='day')
    with pytest.raises(ValueError,match='scene groups'):
        collect_dataset(tmp_path/'dataset',groups=3)


def fake_run(self,recipe,seed):
    result=ep(1,8)
    from opensk.rl.dataset import recipe_features
    v=int(abs(recipe_features(recipe).sum())*10)%255
    rgb=np.full((1,8,128,64,3),v,np.uint8)
    return result._replace(rgb=rgb),dict(frame_times=np.arange(8)/30,deck_mask=np.zeros((8,128,64),bool),
                                        obstacle_pixels=np.full(8,20),initial_xy=np.zeros(2),variation_seed=np.asarray(seed))


def test_resume_after_committed_shard_and_orphan(tmp_path,monkeypatch):
    from opensk.rl.classic import ClassicEnv
    monkeypatch.setattr(ClassicEnv,'run_recipe',fake_run)
    with pytest.raises(InterruptedError):
        collect_dataset(tmp_path,groups=4,stop_after=1)
    shard=next(tmp_path.rglob('*.npz')); digest=store.file_hash(shard)
    # Crash between shard rename and manifest commit: resume adopts verified shard.
    manifest=json.loads((tmp_path/'manifest.json').read_text())
    manifest['completed']={}
    store.atomic_json(tmp_path/'manifest.json',manifest)
    (tmp_path/'crashed.partial').write_bytes(b'incomplete')
    r=collect_dataset(tmp_path,groups=4)
    assert r['new_shards']==35 and r['incomplete_files']==['crashed.partial']
    assert store.file_hash(shard)==digest
    assert audit(tmp_path)['shards']==36
    assert collect_dataset(tmp_path,groups=4)['new_shards']==0
    with pytest.raises(ValueError,match='resume config'):
        collect_dataset(tmp_path,groups=5)
    manifest=json.loads((tmp_path/'manifest.json').read_text())
    manifest['items'][0]['path']='../../escape.npz'
    with pytest.raises(ValueError,match='manifest'): validate_manifest(manifest)


def test_apple_warp_guard_never_imports_or_compiles(monkeypatch):
    from opensk.rl import backend
    monkeypatch.setattr(platform,'system',lambda:'Darwin')
    monkeypatch.setattr(backend.importlib.util,'find_spec',lambda n:pytest.fail('guard imported Warp'))
    assert backend.select_backend('auto')=='classic'
    with pytest.raises(RuntimeError,match='confirmed native crash'):
        backend.select_backend('warp')
    env=object.__new__(__import__('opensk.rl.env',fromlist=['GestureEnv']).GestureEnv)
    with pytest.raises(RuntimeError,match='macOS'): env._build_pixel()
