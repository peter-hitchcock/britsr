# 12.6.21 For this in hddm just using to sim, then r_pr_hddm.py for optimizing 
# 9.3.21 Switched from ipynb because wonky in my VS Code installation
ppcs = 0
pr = 1
t_regressor = 0

import os
import random
from importlib import reload as ir

import pandas as pd
import numpy as np
from kabuki.hierarchical import Knode
import hddm
import pymc as pm
import pymc.progressbar as pbar
from hddm import utils
from py_modules.utilities import gen_rand_str

use_sv = 1
use_st = 1

def DfSubset(df, string):
    return df[df.columns.to_series()[df.columns.to_series().str.contains(string)]]


def DoSimsForPr(
                    # Main effect traces 
                    this_a, this_z, this_t, 
                    # Regressor traces 
                    this_vi, this_vs, this_vv, this_vsv, 
                    # Covariates 
                    this_sess, this_val,
                    use_sv, use_st,
                    sv=0, st=0,
                    ):
    
    # Construct v based on trace and covariates this trial 
    tmp_v = np.NaN
    tmp_v = this_vi
    # Add main effects 
    tmp_v += this_sess * this_vs
    tmp_v += this_val * this_vv
    # Interaction 
    tmp_v += this_val * this_sess * this_vsv

    conv_z = inv_logit(this_z)

    if not use_sv:
        level1 = {'v': tmp_v, 'a': this_a, 't':this_t, 'sv': 0, 'z': conv_z, 'sz': 0, 'st': 0}
    elif not use_st: 
        level1 = {'v': tmp_v, 'a': this_a, 't':this_t, 'sv': sv, 'z': conv_z, 'sz': 0, 'st': 0}
    else: # use both 
        level1 = {'v': tmp_v, 'a': this_a, 't':this_t, 'sv': sv, 'z': conv_z, 'sz': 0, 'st': st}
    out = hddm.generate.gen_rand_data({'level1': level1},
                                                    size=1,
                                                    subjs=1)


    rt = out[0].rt[0]
    response = out[0].response[0]
    print(rt)
    print(response)

    return rt, response


# %%
def inv_logit(p): 
    return np.exp(p) / (1 + np.exp(p))


# Prep #
# %%
sdf = pd.read_csv("../data/cleaned_files/s_bdf.csv")
sf = sdf[["subj_idx", "sess_ctr", "val_ctr"]]


# TEMPORARY : testing out sv  
#t1 = pd.read_csv("../model_res/traces/ALTGAMMASPEC_GR_run_also8k_ddm_add_trialwise_SUBJ-SV_s_vt_poutlier02999.csv")

t1 = pd.read_csv("../model_res/final_traces_and_dics/traces/GR_run_also8k_ddm_add_trialwise_NO-SZ_s_vt_poutlier052177.csv")

# %%
# Get unique IDs 
zn = list(t1.columns[t1.columns.str.contains('z_')])
zs = zn[2:len(zn)]
ids = []
for i in range(len(zs)):
    ids.append(zs[i].split('.')[1])  

# %%
# Subject only df 
t1s = DfSubset(t1, '_subj')
# Parameter dfs 
a_d = t1s.iloc[:, 0:96]
z_d = t1s[t1s.columns.to_series()[t1s.columns.to_series().str.contains('z_')]]
v_int_d = t1s[t1s.columns.to_series()[t1s.columns.to_series().str.contains('v_Intercept')]]
v_val_d = t1s[t1s.columns.to_series()[t1s.columns.to_series().str.contains('v_val_ctr_')]]
v_s_d = t1s[t1s.columns.to_series()[t1s.columns.to_series().str.contains('v_sess_ctr_')]]
v_sv_d = t1s[t1s.columns.to_series()[t1s.columns.to_series().str.contains('v_val_ctr:sess_ctr')]]
t_d = t1s[t1s.columns.to_series()[t1s.columns.to_series().str.contains('t_')]].iloc[:, 0:96]
# These are just at group level 
sv_d = t1["sv"]
st_d = t1["st"]

# # Parameter Recovery
if pr == 1:
# %%
# Find 10th percentile of min and 90th of max in subject traces
    def GenRange(s_traces):
        return [np.quantile(np.min(s_traces), .1), np.quantile(np.max(s_traces), .9)]


    # %%
    # Find some ranges in which to simulate from empirical data
    a_r = GenRange(a_d)
    t_r = GenRange(t_d)
    vi_r = GenRange(v_int_d)
    vv_r = GenRange(v_val_d)
    vs_r = GenRange(v_s_d)
    vsv_r = GenRange(v_sv_d)
    z_r = GenRange(z_d)
    sv_r = GenRange(sv_d)
    st_r = GenRange(st_d)

    # %%
    pr_sims = []

    # Pull subjects for their task contingencies
    for subj in range(len(ids)):
        
        # This subject's info 
        this_subj = sf[sf["subj_idx"] ==  np.float64(ids[subj])]
        
        a_np = random.uniform(a_r[0], a_r[1])
        t_np = random.uniform(t_r[0], t_r[1])
        vi_np = random.uniform(vi_r[0], vi_r[1])
        vv_np = random.uniform(vv_r[0], vv_r[1])
        vs_np = random.uniform(vs_r[0], vs_r[1])
        vsv_np = random.uniform(vsv_r[0], vsv_r[1])
        z_np = random.uniform(z_r[0], z_r[1])
        sv_np = random.uniform(sv_r[0], sv_r[1])
        st_np = random.uniform(st_r[0], st_r[1])

        # Vectorize variables we need 
        val = this_subj["val_ctr"].to_numpy()
        sess = this_subj["sess_ctr"].to_numpy()
        # Preallocate 
        subj_rts, subj_responses = [], []

        
        for r in range(sess.size):
            
            # Pull out trial covariate 
            this_sess = sess[r]
            this_val = val[r]

            sim_rts, sim_responses = DoSimsForPr(
                                        a_np, z_np, t_np,
                                        vi_np, vs_np, vv_np, vsv_np, 
                                        # this_a, this_z, this_alpha, this_t, 
                                        # this_vi, this_vs, this_vv, this_vsv, 
                                        this_sess, this_val,
                                        use_sv, use_st, 
                                        sv=sv_np, st=st_np,
                                        )
            print(sim_rts)
            print(sim_responses)                                        
                                     
            subj_rts.append(sim_rts)
            subj_responses.append(subj_responses)    
            
            sim_for_pr = pd.DataFrame({
                    'subj_idx': pd.Series(np.float64(ids[subj])),
                    'sess_ctr': pd.Series(np.float64(this_sess)),
                    'val_ctr': pd.Series(np.float64(this_val)),
                    'rt': sim_rts, #pd.Series(sim_rts[0]), 
                    'response': sim_responses,#pd.Series(sim_responses[0]),
                    'sim_a': a_np,
                    'sim_t': t_np, #np.repeat(t_np, len(val)),
                    'sim_vi': vi_np, #np.repeat(vi_np, len(val)),
                    'sim_vv': vv_np, #np.repeat(vv_np, len(val)),
                    'sim_vs': vs_np, #np.repeat(vs_np, len(val)),
                    'sim_vsv': vsv_np, #np.repeat(vsv_np, len(val)),
                    'sim_z': inv_logit(z_np),#np.repeat(inv_logit_bounded(z_np), len(val)),
                    'sim_sv': sv_np,
                    'sim_st': st_np,
                })
            pr_sims.append(sim_for_pr)


    # %%
    sims = pd.concat(pr_sims)

    #sims.to_csv("./../model_res/par_recov/HDDM_sims_for_pr_w_sv_" + gen_rand_str() + ".csv")
    sims.to_csv("./../model_res/par_recov/HDDM_sims_for_pr_add_trialwise_NO-SZ_s_vt_poutlier05" + gen_rand_str() + ".csv")

