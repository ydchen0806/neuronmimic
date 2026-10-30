"""P0 repaired intervention contracts; fixture only, no biological claims."""
from __future__ import annotations
from dataclasses import replace
from itertools import combinations
import json
import numpy as np
import pandas as pd
import torch
from .teacher import simulate_teacher, make_stimulus_panel, Stimulus


def chain(n):
    a=np.zeros((n,n));i=np.arange(n-1);a[i,i+1]=a[i+1,i]=1
    return a


def parameterized_events(amplitudes, stimulus, config):
    """Same amplitudes at SAME declared times/sites as the Teacher."""
    steps=int(round(config.duration_ms/config.dt_ms))
    basis=amplitudes.new_zeros((len(amplitudes),steps,config.n_compartments))
    for j,(onset,site) in enumerate(zip(stimulus.onset_ms,stimulus.synapse_compartments)):
        for t in range(steps):
            if abs(t*config.dt_ms-onset)<=0.5*config.dt_ms:
                basis[j,t,site]=1
    return torch.einsum('j,jtn->tn',amplitudes,basis)


def gradient_check(config, model, *, stimulus=None, time_ms=35., eps=1e-3):
    stimulus=stimulus or make_stimulus_panel(config,seed=20261030)[-1]
    step=int(round(time_ms/config.dt_ms));n=config.n_compartments
    if not 0<=step<=round(config.duration_ms/config.dt_ms): raise ValueError('readout outside simulation')
    fd=[];fd_half=[]
    base=simulate_teacher(config,stimulus,adjacency=chain(n))
    for epsilon,values in [(eps,fd),(eps/2,fd_half)]:
        for j in range(len(stimulus.amplitude)):
            aa=list(stimulus.amplitude);bb=list(aa);aa[j]+=epsilon;bb[j]-=epsilon
            plus=simulate_teacher(config,replace(stimulus,amplitude=tuple(aa)),adjacency=chain(n))
            minus=simulate_teacher(config,replace(stimulus,amplitude=tuple(bb)),adjacency=chain(n))
            values.append((plus['voltage'][step,-1]-minus['voltage'][step,-1])/(2*epsilon))
    dtype=model.adjacency.dtype if hasattr(model,'adjacency') else next(model.parameters()).dtype
    amp=torch.tensor(stimulus.amplitude,dtype=dtype,requires_grad=True)
    event=parameterized_events(amp,stimulus,config)
    state=torch.tensor(np.r_[np.full(n,config.resting_voltage),np.zeros(n)][None],dtype=dtype)
    pred=model.rollout(state,event)
    grad=torch.autograd.grad(pred[step,0,n-1],amp)[0].detach().numpy()
    fd=np.asarray(fd);den=np.linalg.norm(fd)*np.linalg.norm(grad)
    return {'model':type(model).__name__,'gradient_cosine':float(fd@grad/den) if den>0 else float('nan'),
            'gradient_relative_l2_error':float(np.linalg.norm(grad-fd)/np.linalg.norm(fd)) if np.linalg.norm(fd)>0 else float('nan'),
            'teacher_gradient':json.dumps(fd.tolist()),'student_gradient':json.dumps(grad.tolist()),
            'fd_half_epsilon_max_difference':float(np.max(np.abs(fd-np.asarray(fd_half)))),
            'gradient_time_ms':step*config.dt_ms,'readout_compartment':n-1,
            'event_sites':json.dumps(list(stimulus.synapse_compartments)),
            'event_times_ms':json.dumps(list(stimulus.onset_ms)),
            'teacher_queries':1+4*len(fd),'base_target':float(base['voltage'][step,-1]),
            'gradient_contract':'specified_event_amplitudes_v2','status':'FIXTURE_TEST_ONLY'}


class _CappedSimplex(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k):
        if not 0<k<x.shape[-1]: raise ValueError('K must be between 0 and n')
        lo=x.min(dim=-1,keepdim=True).values-1
        hi=x.max(dim=-1,keepdim=True).values
        for _ in range(60):
            mid=(lo+hi)/2
            too_large=(x-mid).clamp(0,1).sum(-1,keepdim=True)>k
            lo=torch.where(too_large,mid,lo);hi=torch.where(too_large,hi,mid)
        out=(x-(lo+hi)/2).clamp(0,1)
        ctx.save_for_backward((out>0)&(out<1))
        return out

    @staticmethod
    def backward(ctx, grad):
        active,=ctx.saved_tensors
        active=active.to(grad.dtype)
        average=(grad*active).sum(-1,keepdim=True)/active.sum(-1,keepdim=True).clamp_min(1)
        return active*(grad-average),None


def capped_simplex(x,k): return _CappedSimplex.apply(x,k)


def voltage_objective(rollout, step, compartment, target):
    """Identical voltage/time/squared-error objective for proposal optimization."""
    return (rollout[step,:,compartment]-target).square()


def active_search(config, model, *, k=2, seed=20261030, budget=25, iterations=40,
                  target=None, time_ms=25., onset_ms=16.):
    n=config.n_compartments
    if not 0<k<n or budget<1: raise ValueError('invalid search budget/K')
    if onset_ms>=time_ms or time_ms>config.duration_ms: raise ValueError('invalid observation window')
    rng=np.random.default_rng(seed)
    target=target or make_stimulus_panel(config,seed=seed)[-1]
    step=int(round(time_ms/config.dt_ms));onset=int(round(onset_ms/config.dt_ms))
    value=float(simulate_teacher(config,target,adjacency=chain(n))['voltage'][step,-1])
    # Geometry-only proposals: graph-distance to distal readout, no Teacher query.
    possible=list(combinations(range(n),k))
    heuristic=sorted(possible,key=lambda ss:(sum(n-1-s for s in ss),ss))
    proposals=[('geometry',i,heuristic[i%len(heuristic)]) for i in range(budget)]
    proposals += [('random',i,tuple(sorted(rng.choice(n,k,replace=False).tolist()))) for i in range(budget)]
    # Batched restarts, identical iteration budget. Relaxation preserves total K
    # and each site's amplitude is bounded in [0,1]; hard projection is exact K.
    logits=torch.tensor(rng.normal(0,.05,(budget,n))+k/n,dtype=torch.float32,requires_grad=True)
    optimizer=torch.optim.Adam([logits],lr=.1)
    initial=torch.zeros(budget,2*n);initial[:,:n]=config.resting_voltage
    selector=torch.zeros(step,budget,1);selector[onset]=1
    for _ in range(iterations):
        weights=capped_simplex(logits,k)
        pred=model.rollout(initial,selector*weights.unsqueeze(0))
        loss=voltage_objective(pred,step,n-1,value).mean()
        if not torch.isfinite(loss): raise ValueError('NUMERICAL_FAILURE in search')
        optimizer.zero_grad()
        logits.grad=torch.autograd.grad(loss,logits)[0]
        optimizer.step()
    weights=capped_simplex(logits,k).detach()
    sites=torch.topk(weights,k,dim=-1).indices.sort(dim=-1).values.tolist()
    proposals += [('gradient_topk',i,tuple(ss)) for i,ss in enumerate(sites)]
    rows=[]
    for method,i,ss in proposals:
        s=Stimulus(f'{method}_{i}',ss,(onset_ms,)*k,(1.,)*k,'search',method)
        actual=float(simulate_teacher(config,s,adjacency=chain(n))['voltage'][step,-1])
        rows.append({'method':method,'proposal_index':i,'sites':';'.join(map(str,ss)),'k':k,
                     'amplitude_per_site':1.,'total_amplitude':float(k),'onset_ms':onset_ms,
                     'target_time_ms':time_ms,'readout_compartment':n-1,'teacher_target_voltage':value,
                     'teacher_candidate_voltage':actual,'teacher_abs_error_to_target':abs(actual-value),
                     'teacher_squared_error_to_target':(actual-value)**2,
                     'teacher_candidate_queries':1,'shared_target_queries':1,
                     'teacher_proposal_queries':0,'teacher_evaluation_budget_per_method':budget,
                     'student_optimization_iterations':iterations if method=='gradient_topk' else 0,
                     'seed':seed,'status':'FIXTURE_TEST_ONLY'})
    result=pd.DataFrame(rows)
    result['best_so_far_teacher_error']=result.groupby('method')['teacher_abs_error_to_target'].cummin()
    return result
