## Parameter recovery after running in hddm_ppcs_and_par_recov.py
n_samples = 2500
n_burns = 1000

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

s = gen_rand_str()

#pr_df = pd.read_csv("../model_res/par_recov/HDDM_sims_for_pr4568.csv")
pr_df = pd.read_csv("../model_res/par_recov/HDDM_sims_for_pr_add_trialwise_NO-SZ_s_vt_poutlier059513.csv") 
print(pr_df)

par_rec = hddm.HDDMRegressor(
            pr_df,
            {'model': 'v ~ 1 + val_ctr*sess_ctr', 'link_func': lambda x: x},
            group_only_regressors=False,
            keep_regressor_trace=True, 
            p_outlier=0.05,
            include=['z', 'st', 'sv'],
            group_only_nodes=['st', 'sz'], 
            )


par_rec.sample(n_samples, n_burns)

par_rec.get_traces().to_csv("../model_res/par_recov/HDDM_par_recov_traces_v-val_ddm_add_trialwise_NO-SZ_s_vt_with-z_" + s + ".csv")
pd.Series(par_rec.dic).to_csv("../model_res/par_recov/HDDM_par_recov_dic_v-val_ddm_add_trialwise_NO-SZ_s_vt_with-z_" + s + ".csv")
                                    
