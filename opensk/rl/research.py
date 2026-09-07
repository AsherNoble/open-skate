"""A small CPU world-model experiment with honest held-out measurements."""
from pathlib import Path
import time
import numpy as np
import cv2

from . import store
from .worldmodel import _shrink
from .experts import read_episodes, split_episodes


def pairs(records, horizon=.2):
    cur, nxt, actions, times, owners = [], [], [], [], []
    for r in records:
        frames = _shrink(r['rgb'][None].astype(np.float32)/255.,8)[0]
        ts=r['times']
        for i,t in enumerate(ts):
            if t < 0:
                continue
            j=int(np.argmin(abs(ts-(t+horizon))))
            if j<=i or abs(ts[j]-t-horizon)>.02 or not r['frame_valid'][i:j+1].all():
                continue
            cur.append(frames[i]); nxt.append(frames[j]); actions.append(r['features'])
            times.append(t); owners.append(r['id'])
    if not cur:
        raise ValueError('no valid temporal pairs in partition')
    return dict(cur=np.asarray(cur),nxt=np.asarray(nxt),act=np.asarray(actions),
                time=np.asarray(times),owners=owners)


class RidgeWorld:
    """Low-rank RGB state + recipe/time -> residual future image.

    PCA and feature normalisation use training observations only. Closed-form
    ridge regression keeps the baseline practical on a CPU and reproducible.
    """
    def __init__(self, train, *, rank=16, use_action=True):
        self.use_action=use_action
        cur=train['cur'].reshape(len(train['cur']),-1)
        self.mean=cur.mean(axis=0)
        _,_,vt=np.linalg.svd(cur-self.mean,full_matrices=False)
        self.basis=vt[:min(rank,len(vt))].T
        self.action_mean=train['act'].mean(axis=0)
        self.action_scale=np.maximum(train['act'].std(axis=0),.05)
        self.train_mean=train['nxt'].mean(axis=0)
        self.x=self.features(train)
        self.y=(train['nxt']-train['cur']).reshape(len(cur),-1)

    def features(self,data):
        cur=data['cur'].reshape(len(data['cur']),-1)
        z=(cur-self.mean)@self.basis
        t=data['time'][:,None]
        parts=[z,t,t*t,np.ones_like(t)]
        if self.use_action:
            a=(data['act']-self.action_mean)/self.action_scale
            # Time interactions let one recipe produce different changes while
            # active and after lift. This is still a deliberately small model.
            parts.extend([a,a*t,a*np.exp(-4*t)])
        return np.concatenate(parts,axis=1)

    def fit(self,ridge):
        self.w=np.linalg.solve(self.x.T@self.x+ridge*np.eye(self.x.shape[1]),self.x.T@self.y)
        return self

    def predict(self,data):
        delta=(self.features(data)@self.w).reshape(data['cur'].shape)
        return np.clip(data['cur']+delta,0,1)

    def save(self,path):
        with store.atomic_file(path) as stream:
            np.savez_compressed(stream,mean=self.mean,basis=self.basis,
                                action_mean=self.action_mean,action_scale=self.action_scale,
                                train_mean=self.train_mean,w=self.w,use_action=self.use_action)

    @classmethod
    def load(cls,path):
        model=object.__new__(cls)
        with np.load(path,allow_pickle=False) as z:
            for key in ('mean','basis','action_mean','action_scale','train_mean','w'):
                setattr(model,key,z[key])
            model.use_action=bool(z['use_action'])
        return model


def mse(a,b):
    return float(np.mean((a-b)**2))


def pixel_scores(model,data,ablated):
    pred=model.predict(data)
    # Domain-balanced ordering repeats gestures; a fixed cyclic shift can
    # accidentally leave every action unchanged. Seeded permutation does not.
    permutation=np.random.default_rng(734).permutation(len(data['act']))
    swapped=dict(data,act=data['act'][permutation])
    other=model.predict(swapped)
    return dict(model_mse=mse(pred,data['nxt']),persistence_mse=mse(data['cur'],data['nxt']),
                mean_frame_mse=mse(model.train_mean,data['nxt']),
                no_action_model_mse=mse(ablated.predict(data),data['nxt']),
                shuffled_action_mse=mse(other,data['nxt']),
                action_prediction_change_mse=mse(pred,other),pairs=len(data['cur']),
                episodes=len(set(data['owners'])))


def paired_effect(model, records, horizon=.2):
    """Same initial scene, different recipe: compare predicted and actual change."""
    groups={}
    for r in records:
        if r['source']=='sim':
            groups.setdefault((r['group'],r['park'],r['appearance']),[]).append(r)
    predicted,actual=[],[]
    for group in groups.values():
        if len(group)<2:
            continue
        group=group[:2]
        frames=[_shrink(r['rgb'][None].astype(np.float32)/255,8)[0] for r in group]
        if not np.array_equal(group[0]['rgb'][0],group[1]['rgb'][0]):
            raise ValueError('counterfactual scenes must have identical initial frames')
        data=dict(cur=np.stack([f[0] for f in frames]),act=np.stack([r['features'] for r in group]),time=np.zeros(2))
        pred=model.predict(data)
        truth=np.stack([f[np.argmin(abs(r['times']-horizon))] for f,r in zip(frames,group)])
        predicted.append(pred[1]-pred[0]);actual.append(truth[1]-truth[0])
    if not predicted:
        return dict(pairs=0)
    p,a=np.asarray(predicted),np.asarray(actual)
    return dict(pairs=len(p),predicted_change_mse=mse(p,0),actual_change_mse=mse(a,0),
                effect_error_mse=mse(p,a),zero_effect_error_mse=mse(a,0),
                effect_cosine=float(np.sum(p*a)/max(1e-12,np.linalg.norm(p)*np.linalg.norm(a))))


OUTCOME_NAMES=('roll_deg','yaw_deg','peak_height_m','air_s','displacement_m')


def outcome_features(records):
    return np.stack([r['features'] for r in records])


def fit_outcomes(train,val):
    train=[r for r in train if r['known']]
    val=[r for r in val if r['known']]
    if not train or not val:
        return None
    x=outcome_features(train); y=np.stack([r['outcomes'] for r in train])
    xm=x.mean(0); xs=np.maximum(x.std(0),.05)
    ym=y.mean(0); ys=np.maximum(y.std(0),.01)
    def feat(a):
        z=(a-xm)/xs
        return np.c_[z,z*z,np.ones(len(z))]
    xx=feat(x)
    vx=feat(outcome_features(val)); vy=np.stack([r['outcomes'] for r in val])
    best=None
    for ridge in (.1,1.,10.,100.):
        w=np.linalg.solve(xx.T@xx+ridge*np.eye(xx.shape[1]),xx.T@((y-ym)/ys))
        loss=mse((vx@w*ys+ym-vy)/ys,0)
        if best is None or loss<best[0]:
            best=(loss,w,ridge)
    return dict(xm=xm,xs=xs,ym=ym,ys=ys,w=best[1],ridge=best[2])


def predict_outcomes(model,features):
    z=(features-model['xm'])/model['xs']
    return np.c_[z,z*z,np.ones(len(z))]@model['w']*model['ys']+model['ym']


def score_outcomes(model,records):
    known=[r for r in records if r['known']]
    if model is None or not known:
        return dict(labelled_episodes=0,reason='no independently measured physical labels')
    pred=predict_outcomes(model,outcome_features(known))
    truth=np.stack([r['outcomes'] for r in known])
    return dict(labelled_episodes=len(known),model_mae=dict(zip(OUTCOME_NAMES,np.abs(pred-truth).mean(0).tolist())),
                mean_baseline_mae=dict(zip(OUTCOME_NAMES,np.abs(model['ym']-truth).mean(0).tolist())))


def search(model, *, seed=9001, candidates=12):
    """Rank new gestures with the learned head, then validate ALL in MuJoCo."""
    if model is None:
        return dict(status='unavailable without physical training labels')
    from .dataset import controlled_actions, recipe_features
    from .action import to_recipe
    from .classic import ClassicEnv
    actions=controlled_actions(seed,candidates)
    recipes=[to_recipe(a) for a in actions]
    features=np.stack([recipe_features(r) for r in recipes])
    predicted=predict_outcomes(model,features)[:,2]
    selected=int(np.argmax(predicted))
    env=ClassicEnv(park='flat',appearance='day',seconds=1.2,vary=True)
    actual=[]
    for recipe in recipes:
        ep,_=env.run_recipe(recipe,seed)
        actual.append(float(ep.peak_height[0]) if ep.valid[0] else None)
    valid=[v for v in actual if v is not None]
    return dict(objective='peak height (metres)',candidates=candidates,seed=seed,
                selected=selected,predicted_height_m=float(predicted[selected]),
                actual_height_m=actual[selected],random_choice_expected_height_m=float(np.mean(valid)),
                oracle_best_height_m=max(valid),all_actual_heights_m=actual,
                all_predicted_heights_m=predicted.tolist(),recipe=recipes[selected],
                device_execution='not performed')


def prediction_sheet(path,model,data):
    indices=np.linspace(0,len(data['cur'])-1,6,dtype=int)
    pred=model.predict(data)
    rows=[]
    for i in indices:
        row=[]
        for frame in (data['cur'][i],data['nxt'][i],pred[i],abs(pred[i]-data['nxt'][i])):
            row.append(cv2.resize(np.uint8(np.clip(frame,0,1)*255),(128,256),interpolation=cv2.INTER_NEAREST))
        rows.append(np.concatenate(row,axis=1))
    rgb=np.concatenate(rows,axis=0)
    ok,png=cv2.imencode('.png',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError('could not encode prediction sheet')
    with store.atomic_file(path) as stream:
        stream.write(png.tobytes())


def experiment(roots,output,*,mode='synthetic',run_search=True):
    start=time.perf_counter()
    output=Path(output)
    records=read_episodes(roots)
    split=split_episodes(records,mode=mode)
    if mode=='synthetic' and any(r['source']=='device' for r in records):
        # Direct transfer measurement: no device frames enter this model's
        # training, PCA, normalisation, or hyperparameter selection.
        split['device_transfer']=split_episodes(records,mode='expert')['test']
    store.atomic_json(output/'split.json',{k:[dict(id=r['id'],group=r['group'],source=r['source']) for r in v]
                                           for k,v in split.items()})
    train=pairs(split['train']); val=pairs(split['validation'])
    models=[]
    for action in (True,False):
        model=RidgeWorld(train,use_action=action)
        losses=[]
        for ridge in (.1,1.,10.,100.):
            model.fit(ridge)
            losses.append(mse(model.predict(val),val['nxt']))
        ridge=(.1,1.,10.,100.)[int(np.argmin(losses))]
        model.fit(ridge)
        models.append((model,ridge))
    model,ridge=models[0]; ablated=models[1][0]
    model.save(output/'world_model.npz')
    outcomes=fit_outcomes(split['train'],split['validation'])
    if outcomes:
        with store.atomic_file(output/'outcome_model.npz') as stream:
            np.savez_compressed(stream,**outcomes)
    report=dict(mode=mode,split_counts={k:len(v) for k,v in split.items()},
                horizon_s=.2,prediction_resolution=[8,16],training_resolution=[64,128],
                model='PCA(16) + ridge residual RGB, recipe/time interactions',ridge=ridge,
                partitions={},physical_outcomes={},limitations=[
                    'Small paired tail-gesture distribution; not full trick coverage.',
                    'Physical outcomes use simulator labels; device outcomes remain unknown.',
                    'No phone execution or measured sim-to-real transfer.',
                    'Jetson Nanos and CUDA throughput have not been benchmarked.'])
    for name in ('validation','test','domain','device_transfer'):
        if not split.get(name):
            continue
        data=pairs(split[name])
        report['partitions'][name]=pixel_scores(model,data,ablated)
        report['physical_outcomes'][name]=score_outcomes(outcomes,split[name])
        prediction_sheet(output/f'{name}_predictions.png',model,data)
        # Report real/synthetic subsets separately; mixed aggregates can hide
        # regressions on the smaller source.
        for source in ('sim','device'):
            subset=[r for r in split[name] if r['source']==source]
            if subset:
                report['partitions'][f'{name}_{source}']=pixel_scores(model,pairs(subset),ablated)
    report['search']=search(outcomes) if run_search else dict(status='disabled')
    report['paired_gesture_effect']=paired_effect(model,split['test'])
    report['elapsed_s']=time.perf_counter()-start
    store.atomic_json(output/'metrics.json',report)
    lines=[f'# Local research baseline: {mode}', '',
           f"Split episodes: {report['split_counts']}. Horizon 0.2 s; model predicts 8×16 RGB from 64×128 captures.", '',
           '| Partition | Model MSE | Persistence | Mean frame | No action | Shuffled action |',
           '|---|---:|---:|---:|---:|---:|']
    for name,s in report['partitions'].items():
        lines.append(f"| {name} | {s['model_mse']:.6f} | {s['persistence_mse']:.6f} | {s['mean_frame_mse']:.6f} | {s['no_action_model_mse']:.6f} | {s['shuffled_action_mse']:.6f} |")
    lines.extend(['','Prediction sheets: current, actual +0.2 s, predicted +0.2 s, absolute error. Pixels are enlarged with nearest-neighbour; they are not high-resolution predictions.','',
                  'Physical MAEs and gesture-search validation are in metrics.json. Model selection uses validation only. Entire scene/capture groups and shared frames stay together.','',
                  *[f'- {s}' for s in report['limitations']]])
    with store.atomic_file(output/'REPORT.md') as stream:
        stream.write(('\n'.join(lines)+'\n').encode())
    return report
