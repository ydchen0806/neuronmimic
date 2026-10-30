"""Scientific regression tests against analytic cases and NumPy Teacher."""
import json
import numpy as np
import pytest
import torch
from neuronmimic.physics_guided import CableTeacherConfig, PhysicsOnlyDynamics, Stimulus, simulate_teacher
from neuronmimic.physics_guided.metrics import evaluate_student
from neuronmimic.physics_guided.experiments import (chain,gradient_check,parameterized_events,
    active_search,capped_simplex,voltage_objective)


class Replay:
    n=1
    def __init__(self,state,voltage_shift=0.,g_shift=0.,step_error=False):
        self.state=torch.tensor(state);self.voltage_shift=voltage_shift;self.g_shift=g_shift;self.step_error=step_error
    def rollout(self,initial,event):
        p=self.state.clone();p[1:,0]+=self.voltage_shift;p[1:,1]+=self.g_shift
        return p[:,None,:]
    def step(self,state,event):
        p=self.state[1:].clone()
        if self.step_error:p[-1,0]+=2.
        p[:,1]+=self.g_shift
        return p


def example():
    state=np.array([[-65.,0.],[-64.,.1],[-63.,.2]],dtype=np.float32)
    return dict(state=state,event=np.zeros((2,1),np.float32),time_ms=np.arange(3.))


def test_perfect_replay_zero_and_numpy_voltage_reference():
    ex=example();m=evaluate_student(Replay(ex['state']),[ex],threshold=-64.5)
    assert m['multi_step_voltage_rmse']==m['one_step_voltage_rmse']==0
    m=evaluate_student(Replay(ex['state'],voltage_shift=2.,g_shift=1000.),[ex],threshold=-64.5)
    assert m['multi_step_voltage_rmse']==2.
    assert m['multi_step_conductance_rmse']>999
    assert m['one_step_voltage_rmse']==0


def test_one_step_includes_late_transitions():
    ex=example();m=evaluate_student(Replay(ex['state'],step_error=True),[ex])
    assert m['one_step_voltage_rmse']==pytest.approx(np.sqrt((0.+4.)/2))


def test_missing_response_counted_not_zero_latency():
    ex=example();m=evaluate_student(Replay(ex['state'],voltage_shift=-10),[ex],threshold=-64.5)
    assert m['response_fn']==1 and m['latency_valid_pairs']==0 and np.isnan(m['latency_error_ms'])
    ex['state'][:,0]=-65
    m=evaluate_student(Replay(ex['state'],voltage_shift=10),[ex],threshold=-64.5)
    assert m['response_fp']==1 and m['graded_response_error']==1


def test_nonfinite_rejected_instead_of_omitted():
    ex=example()
    with pytest.raises(ValueError,match='NUMERICAL_FAILURE'):
        evaluate_student(Replay(ex['state'],voltage_shift=float('nan')),[ex])


def test_exact_physics_against_independent_numpy_teacher():
    cfg=CableTeacherConfig(n_compartments=4,duration_ms=5,dt_ms=.05)
    stim=Stimulus('test',(0,3),(.5,2.),(.8,.4),'test','fixture')
    tr=simulate_teacher(cfg,stim)
    model=PhysicsOnlyDynamics(torch.tensor(chain(4)),cfg).double()
    state=torch.tensor(np.concatenate([tr['voltage'],tr['conductance']],axis=1))
    event=torch.tensor(tr['event'])
    pred=model.rollout(state[:1],event)[:,0].detach().numpy()
    np.testing.assert_allclose(pred,state.numpy(),atol=2e-6,rtol=1e-7)


def test_gradient_same_event_and_future_causality():
    cfg=CableTeacherConfig(n_compartments=4,duration_ms=5)
    stim=Stimulus('test',(0,3),(.5,4.),(.8,.4),'test','fixture')
    model=PhysicsOnlyDynamics(torch.tensor(chain(4)),cfg).double()
    result=gradient_check(cfg,model,stimulus=stim,time_ms=3.)
    fd=np.array(json.loads(result['teacher_gradient']));grad=np.array(json.loads(result['student_gradient']))
    np.testing.assert_allclose(grad,fd,rtol=2e-5,atol=1e-6)
    assert fd[1]==grad[1]==0
    assert result['gradient_cosine']>.99999
    amp=torch.tensor(stim.amplitude,dtype=torch.float64)
    np.testing.assert_array_equal(parameterized_events(amp,stim,cfg).numpy(),simulate_teacher(cfg,stim)['event'])


def test_capped_simplex_budget_and_derivative():
    x=torch.tensor([[.25,.3,.5,.65]],dtype=torch.float64,requires_grad=True)
    y=capped_simplex(x,2)
    assert torch.all((y>=0)&(y<=1)) and float(y.sum().detach())==pytest.approx(2.)
    assert torch.autograd.gradcheck(lambda a:capped_simplex(a,2),(x,),eps=1e-6,atol=1e-5)


def test_voltage_objective_ignores_conductance_and_other_times():
    x=torch.zeros((3,1,4));x[1,0,1]=3.;x[-1,0,-1]=1000
    assert voltage_objective(x,1,1,1.).item()==4.


def test_real_search_exact_k_equal_budget_and_teacher_reference():
    cfg=CableTeacherConfig(n_compartments=4,duration_ms=5)
    target=Stimulus('target',(0,),(.5,),(1.,),'test','fixture')
    model=PhysicsOnlyDynamics(torch.tensor(chain(4)).float(),cfg)
    result=active_search(cfg,model,k=2,budget=3,iterations=2,target=target,time_ms=3.,onset_ms=1.)
    assert result.groupby('method').size().eq(3).all()
    for r in result.itertuples():
        sites=tuple(map(int,r.sites.split(';')))
        assert len(sites)==len(set(sites))==2 and r.total_amplitude==2
        trace=simulate_teacher(cfg,Stimulus('reference',sites,(1.,1.),(1.,1.),'test','fixture'))
        value=trace['voltage'][60,-1]
        assert value==r.teacher_candidate_voltage
        assert r.teacher_squared_error_to_target==pytest.approx((value-r.teacher_target_voltage)**2)
