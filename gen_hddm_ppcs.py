## Script for generating PPCs via HDDM since having difficulty simulating with R weiner using st ##
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

individual_sv = 0
regr_sv = 0
use_sv = 1
use_z = 1
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
                    use_sv, sv=0,
                    use_st=0, st=0, use_sv_r=0, sv_vv=0,
                    idx=0,
                    ):
    
    #print("IDX", idx)
    if use_sv_r:
        tmp_sv = np.NaN
        tmp_sv = sv
        tmp_sv += this_val * sv_vv
    else:
        tmp_sv = sv

    # Construct v based on trace and covariates this trial 
    tmp_v = np.NaN
    tmp_v = this_vi
    # Add main effects 
    tmp_v += this_sess * this_vs
    tmp_v += this_val * this_vv
    # Interaction 
    tmp_v += this_val * this_sess * this_vsv

    conv_z = inv_logit(this_z)

    if use_sv and not use_st:
        level1 = {'v': tmp_v, 'a': this_a, 't': this_t, 'sv': tmp_sv, 'z': conv_z, 'sz': 0, 'st': 0}
    elif use_st:
        level1 = {'v': tmp_v, 'a': this_a, 't': this_t, 'sv': tmp_sv, 'z': conv_z, 'sz': 0, 'st': st}
    else: 
        level1 = {'v': tmp_v, 'a': this_a, 't': this_t, 'sv': 0, 'z': conv_z, 'sz': 0, 'st': 0}

    #print("st", st)
    out = hddm.generate.gen_rand_data({'level1': level1},
                                                    size=1,
                                                    subjs=1)

    # data_a, params_a = hddm.generate.gen_rand_data({'level1': level1},
    #                                             size=1,
    #                                             subjs=1)


    rt = out[0].rt[0]
    response = out[0].response[0]
    
    return rt, response


# %%
def inv_logit(p): 
    return np.exp(p) / (1 + np.exp(p))


# Prep #
# %%
sdf = pd.read_csv("../data/cleaned_files/s_bdf.csv")

sf = sdf[["subj_idx", "sess_ctr", "val_ctr"]]
 
#t1 = pd.read_csv("../model_res/traces/s_vt_sv-val_poutlier053596.csv")
#t1 = pd.read_csv("../model_res/traces/s_vt_sv-val_plus-st_poutlier059733.csv")
t1 = pd.read_csv("../model_res/final_traces_and_dics/traces/GR_run_also8k_ddm_add_trialwise_NO-SZ_s_vt_poutlier052177.csv")
# %%
# Get unique IDs 
zn = list(t1.columns[t1.columns.str.contains('z_')])
zs = zn[2:len(zn)]

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
if individual_sv:
    sv_d = t1s[t1s.columns.to_series()[t1s.columns.to_series().str.contains('sv_')]]
elif regr_sv:
    sv_int_d_group = t1['sv(-0.9943157594653141)']
    sv_val_d_group = t1['sv(1.00567346803657)']
else: 
    group_sv = t1["sv"]

ids = []
an = list(a_d.columns)
for i in range(len(an)):
    ids.append(an[i].split('.')[1])
    

# %%
pr_sims = []

if use_st:
    group_st = t1["st"]


# Pull subjects for their task contingencies
for subj in range(len(ids)):
    
    # This subject's info 
    this_subj = sf[sf["subj_idx"] ==  np.float64(ids[subj])]
    print(this_subj)

    ppc_indices = random.sample(range(0, len(a_d)), 25)
    
    # Pull sample traces for this subj and vectorize
    a_np = a_d.iloc[ppc_indices, subj].to_numpy()
    t_np = t_d.iloc[ppc_indices, subj].to_numpy()
    vi_np = v_int_d.iloc[ppc_indices, subj].to_numpy()
    vv_np = v_val_d.iloc[ppc_indices, subj].to_numpy()
    vs_np = v_s_d.iloc[ppc_indices, subj].to_numpy()
    vsv_np = v_sv_d.iloc[ppc_indices, subj].to_numpy()
    if use_z:
        z_np = z_d.iloc[ppc_indices, subj].to_numpy()
    else:
        z_np = [.5 for i in range(len(ppc_indices))]
    if use_sv:
        if individual_sv:
            sv_np = sv_d.iloc[ppc_indices, subj].to_numpy()
        elif regr_sv:
            sv_np = sv_int_d_group.iloc[ppc_indices].to_numpy()    
            sv_vv_all = sv_val_d_group[ppc_indices].to_numpy()    
        else:
            sv_np = group_sv.iloc[ppc_indices].to_numpy()    
    if use_st:
        # Group level only
        st_np_g = group_st.iloc[ppc_indices].to_numpy()
    # Not actually used if using st so this is just a filler for function
    else: 
        st_np_g = [.5 for i in range(len(ppc_indices))]
    # Vectorize variables we need 
    val = this_subj["val_ctr"].to_numpy()
    sess = this_subj["sess_ctr"].to_numpy()
    # Preallocate 
    subj_rts, subj_responses = [], []

    # Number of trace samples 
    for pp in range(len(ppc_indices)):
    
        # Generate one sim for every PPC 
        for r in range(sess.size):
            
            # # Pull out trial covariate 
            sim_rts, sim_responses = DoSimsForPr(
                                        # Pull the appropriate trace sample 
                                        a_np[pp], z_np[pp], t_np[pp],
                                        vi_np[pp], vs_np[pp], vv_np[pp], vsv_np[pp], 
                                        sess[r], val[r],
                                        use_sv, 
                                        sv_np[pp],
                                        use_st, st_np_g[pp],
                                        #regr_sv, sv_vv_all[pp],
                                        pp
                                    )
            print(sim_rts)
            print(sim_responses)                                        
                                        
            subj_rts.append(sim_rts)
            subj_responses.append(subj_responses)    
            
            sim_for_pr = pd.DataFrame({
                    'subj_idx': pd.Series(np.float64(ids[subj])),
                    'sess_ctr': pd.Series(np.float64(sess[r])),
                    'val_ctr': pd.Series(np.float64(val[r])),
                    'rt': sim_rts, #pd.Series(sim_rts[0]), 
                    'response': sim_responses,#pd.Series(sim_responses[0]),
                    # 'sim_a': a_np[pp],
                    # 'sim_t': t_np[pp], #np.repeat(t_np, len(val)),
                    # 'sim_vi': vi_np[pp], #np.repeat(vi_np, len(val)),
                    # 'sim_vv': vv_np[pp], #np.repeat(vv_np, len(val)),
                    # 'sim_vs': vs_np[pp], #np.repeat(vs_np, len(val)),
                    # 'sim_vsv': vsv_np[pp], #np.repeat(vsv_np, len(val)),
                    # 'sim_z': inv_logit(z_np[pp]),#np.repeat(inv_logit_bounded(z_np), len(val)),
                    # 'sim_sv': sv_np[pp],
                })
            pr_sims.append(sim_for_pr)


# %%
sims = pd.concat(pr_sims)

# %%

#sims.to_csv("./../model_res/ppcs/HDDM_ppcs_v-val_OUT-RMed_sv-val_and_w-st-grp-and-z_" + gen_rand_str() + ".csv")
sims.to_csv("./../model_res/ppcs/HDDM_ppcs_v-val_ddm_add_trialwise_NO-SZ_s_vt_with-z_" + gen_rand_str() + ".csv")

