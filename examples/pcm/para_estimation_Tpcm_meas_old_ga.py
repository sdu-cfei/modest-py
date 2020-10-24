# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 12:44:04 2020
@author: taoy
"""
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

import modestpy


    # Paths
ms_file = os.path.join('Tair.csv')
ideal_file=os.path.join('Tpcm_measurement.txt')
fmu_file = os.path.join('para_est.fmu')
res_dir = os.path.join('results_test_modestga','old_ga')

if not os.path.exists(res_dir):
    os.makedirs(res_dir)

# Training and validation periods
trn_t0 = 0 * 86400
trn_t1 = trn_t0 + 3 * 86400  
vld_t0 = trn_t1
vld_t1 = vld_t0 + 2* 86400 - 60

# Read measurements
ms = pd.read_csv(ms_file,sep=";", header=None)
ms.index=ms.index*60
ms=ms[ms.index<432000]
ideal = pd.read_csv(ideal_file,sep="\t", header=None,index_col=0)
ideal.index=ideal.index*60
#ideal.index=ideal.index*60
ideal[1]=ideal[1]+273.15
ideal=ideal[ideal.index<432000]


# Assign model inputs and desired outputs
inp = pd.DataFrame(ms[0])
#inp['time'] = (inp.index - inp.index[0]).total_seconds()  # ModestPy needs index in seconds
inp.index.name='time'                                       # ModestPy needs index named 'time'
inp.columns=['Tair']
ideal.index.name='time'
ideal.columns=['Tpcm']
                     
inp.to_csv(os.path.join(res_dir, 'inp.csv'))
ideal.to_csv(os.path.join(res_dir, 'ideal.csv'))

ax = inp.loc[trn_t0:trn_t1].plot(subplots=True)
fig = ax[0].get_figure()
fig.savefig(os.path.join(res_dir, 'inp_training.png'), dpi=200)

ax = inp.loc[vld_t0:vld_t1].plot(subplots=True)
fig = ax[0].get_figure()
fig.savefig(os.path.join(res_dir, 'inp_validation.png'), dpi=200)

ax = ideal.loc[trn_t0:trn_t1].plot(subplots=True)
fig = ax[0].get_figure()
fig.savefig(os.path.join(res_dir, 'ideal_training.png'), dpi=200)

ax = ideal.loc[vld_t0:vld_t1].plot(subplots=True)
fig = ax[0].get_figure()
fig.savefig(os.path.join(res_dir, 'ideal_validation.png'), dpi=200)

# Parameters
known = dict()

est = dict()
#est['pcm.convection_coefficient1.k_a']= (0.025,0.02,0.03)
#est['pcm.heatCapacitor_PCM.hf_pcm']=(14000.,13500.,14500.)
est['pcm.thermalConductor.G']=(8.,5.,15.)
est['pcm.Lf.Tmc']=(19+273.15,18+273.15,22+273.15)
est['pcm.Lf.Tmh']=(22+273.15,18+273.15,24+273.15)


ic_param = dict()

ic_param['pcm.heatCapacitor_PCM.T'] = 'Tpcm'

#ic_param['co2.balance.CO2ppmv_i'] = 'CO2'

    # Estimation
ga_opts = {'maxiter': 40, 'tol': 1e-7, 'lhs': True, 'pop_size': 30}
#scipy_opts = {
#        'solver': 'L-BFGS-B',
#        'options': {'maxiter': 50, 'tol': 1e-12}
#        }

session = modestpy.Estimation(res_dir, fmu_file, inp, known, est, ideal,
        lp_n=3, lp_frame=(trn_t0, trn_t1),
        vp=(vld_t0, vld_t1),
        methods=('GA', ),
        ga_opts=ga_opts, 
        ic_param=ic_param, ftype='NRMSE', seed=1)

t0=time.time()
estimates = session.estimate()

cputime = int(time.time() - t0)
with open(os.path.join(res_dir, 'cputime.txt'), 'w') as f:
            f.write("{}\n".format(cputime))

vld = session.validate()
vld_err = vld[0]
vld_res = vld[1]

with open(os.path.join(res_dir, 'vld_err.txt'), 'w') as f:
        for k in vld_err:
            f.write("{}: {:.5f}\n".format(k, vld_err[k]))

vld_res.to_csv(os.path.join(res_dir, 'vld_res.csv'))

    # Save all parameters
parameters = pd.DataFrame(index=[0])
for k in estimates:
    parameters[k] = estimates[k]
for k in known:
    parameters[k] = known[k]

    # Remove ic params (if present, that's probably because `known` has been modified within ModestPy)
for p in ic_param:
    if p in list(parameters.columns):
        parameters = parameters.drop(p, axis=1)

parameters.to_csv(os.path.join(res_dir, 'parameters.csv'), index=False)
