"""Aligned, channel-specific evaluation. Historical v1 metrics are invalidated."""
from __future__ import annotations
import time
from typing import Iterable
import numpy as np
import torch


def latency_ms(trace, time_ms, threshold, *, readout_compartment=None):
    signal = trace[:, int(readout_compartment)] if readout_compartment is not None else np.max(trace, axis=-1)
    idx = np.flatnonzero(signal > threshold)
    return float(time_ms[idx[0]]) if len(idx) else float('nan')


def evaluate_student(model, examples: Iterable[dict], *, threshold=-45.0, readout_compartment=None):
    """Pooled RMSE over aligned samples; latency conditional on BOTH crossings.

    Nonfinite states invalidate evaluation rather than disappearing in nanmean.
    State order is [voltage (mV), conductance (fixture units)].
    """
    sse = dict(one_v=0., multi_v=0., one_g=0., multi_g=0.)
    count = 0
    latency, response_errors = [], []
    tp=fp=fn=tn=negative_g=total_g=0
    teacher_responses=student_responses=0
    elapsed=0.
    n_examples=0
    for ex in examples:
        state=torch.from_numpy(ex['state']).float()
        event=torch.from_numpy(ex['event']).float()
        if state.ndim != 2 or state.shape[1] != 2*model.n or event.shape != (len(state)-1, model.n):
            raise ValueError('unaligned state/event dimensions')
        if not torch.isfinite(state).all() or not torch.isfinite(event).all():
            raise ValueError('nonfinite reference input')
        with torch.no_grad():
            start=time.perf_counter()
            raw=model.rollout(state[:1], event)
            elapsed += time.perf_counter()-start
            one=model.step(state[:-1], event)
        if raw.shape != (len(state),1,2*model.n) or one.shape != state[1:].shape:
            raise ValueError('Student returned invalid state dimensions')
        pred=raw[:,0,:]  # explicit alignment; never broadcast across time
        if not torch.isfinite(pred).all() or not torch.isfinite(one).all():
            raise ValueError('NUMERICAL_FAILURE: nonfinite Student prediction')
        for prefix, prediction in [('one',one),('multi',pred[1:])]:
            for channel,sl in [('v',slice(0,model.n)),('g',slice(model.n,None))]:
                error=prediction[:,sl].double()-state[1:,sl].double()
                sse[prefix+'_'+channel] += float(error.square().sum())
        count += (len(state)-1)*model.n
        t=np.asarray(ex['time_ms'])
        if t.shape != (len(state),) or not np.all(np.diff(t)>0):
            raise ValueError('invalid time axis')
        a=latency_ms(pred[:,:model.n].numpy(),t,threshold,readout_compartment=readout_compartment)
        b=latency_ms(state[:,:model.n].numpy(),t,threshold,readout_compartment=readout_compartment)
        pa,pb=np.isfinite(a),np.isfinite(b)
        tp+=int(pa and pb);fp+=int(pa and not pb);fn+=int(pb and not pa);tn+=int(not pa and not pb)
        teacher_responses+=int(pb);student_responses+=int(pa)
        if pa and pb: latency.append(abs(a-b))
        response_errors.append(pa != pb)
        negative_g+=int((pred[1:,model.n:]<0).sum());total_g+=pred[1:,model.n:].numel()
        n_examples+=1
    if not count: raise ValueError('empty evaluation set')
    return {'one_step_voltage_rmse':float(np.sqrt(sse['one_v']/count)),
            'multi_step_voltage_rmse':float(np.sqrt(sse['multi_v']/count)),
            'one_step_conductance_rmse':float(np.sqrt(sse['one_g']/count)),
            'multi_step_conductance_rmse':float(np.sqrt(sse['multi_g']/count)),
            'latency_error_ms':float(np.mean(latency)) if latency else float('nan'),
            'latency_valid_pairs':len(latency),'n_test_stimuli':n_examples,
            'teacher_response_count':teacher_responses,'student_response_count':student_responses,
            'response_tp':tp,'response_fp':fp,'response_fn':fn,'response_tn':tn,
            'graded_response_error':float(np.mean(response_errors)),
            'negative_conductance_fraction':negative_g/total_g,
            'finite_fraction':1.,'rollout_wall_clock_s':elapsed,
            'metric_version':'aligned_voltage_v2','voltage_sample_count':count}
