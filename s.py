# To start from shell locally, may need to run $ conda init then restart terminal, then: 
# $ chmod +x ./r2_local.sh   
# $ ./r2_local.sh 
# or use r4local for modified hddm (hddmacrosstrial) that allows subject-wise sv  (also may need to activate env in terminal before)  

cluster_run = 0
split_half = 0
odd = 0

test_retest = 1
pre = 0
which_model = 0
older = 0
n_samples = 2500
n_burns = 1000
for_final = 0

# cluster_run=0 for local, 1 for cluster run. 
# which_model allows running one model at a time on cluster
# ^ INITIALIZATIONS THAT WILL BE CHANGED ON CLUSTER
#---- MINIMAL SCRIPT FOR RUNNING HDDM FULL MODELS ON CLUSTER -------------#
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
# from py_modules.utilities import run_model_save_outs
###########################################################################
sdf = hddm.load_csv("./../data/cleaned_files/s_bdf.csv")
pre_sdf = sdf[sdf.session=="Pre"]
post_sdf = sdf[sdf.session=="Post"]
even_sdf = hddm.load_csv("./../data/cleaned_files/s_bdf_even.csv")
odd_sdf = hddm.load_csv("./../data/cleaned_files/s_bdf_odd.csv")

tvs = {"model": "t ~ val_ctr*sess_ctr", 'link_func': lambda x:x}
vs = {"model": "v ~ val_ctr*sess_ctr", 'link_func': lambda x:x}

s = gen_rand_str()


if for_final == 1:
    ## 2.22 - Simplified/oganized code used to produce key results ##

    ## Baseline model  
    # mb = hddm.HDDM(
    #             sdf,
    #             p_outlier=.05,
    #             group_only_nodes=['sv', 'st', 'sz'], 
    #             )


    # mb.find_starting_values()
    # mb.sample(n_samples, n_burns)

    # mb.get_traces().to_csv('./../model_res/traces/s_b_389.csv')
    # pd.Series(mb.dic).to_csv('./../model_res/dic/s_b_389_dic.csv')

    ## Just valence 
    # v = {"model": "v ~ val_ctr", 'link_func': lambda x:x}

    # mb1 =  hddm.HDDMRegressor(
    #             sdf,
    #             v,
    #             group_only_regressors=False,
    #             keep_regressor_trace=True, 
    #             p_outlier=.05,
    #             group_only_nodes=['sv', 'st', 'sz'], 
    #             )


    # mb1.find_starting_values()
    # mb1.sample(n_samples, n_burns)

    # mb1.get_traces().to_csv('./../model_res/traces/s_b1_3987.csv')
    # pd.Series(mb1.dic).to_csv('./../model_res/dic/s_b1_3987_dic.csv')


    ## Val * sess 
    # vs = {"model": "v ~ val_ctr*sess_ctr", 'link_func': lambda x:x}
    # m1 = hddm.HDDMRegressor(
    #             sdf,
    #             vs,
    #             group_only_regressors=False,
    #             keep_regressor_trace=True, 
    #             p_outlier=.05,
    #             group_only_nodes=['sv', 'st', 'sz'], 
    #             )


    # m1.find_starting_values()
    # m1.sample(n_samples, n_burns)

    # m1.get_traces().to_csv('./../model_res/traces/s_vt.csv')
    # pd.Series(m1.dic).to_csv('./../model_res/dic/s_vt_dic.csv')

    ## Begin rerun with p = .05  
    # Add z 
    #print("**New run 1**")
    # m1 = hddm.HDDMRegressor(
    #             sdf,
    #             vs,
    #             group_only_regressors=False,
    #             keep_regressor_trace=True, 
    #             p_outlier=0.05,
    #             include=['z'],
    #             group_only_nodes=['sv', 'st', 'sz'], 
    #             )

    # m1.find_starting_values()
    # m1.sample(n_samples, n_burns)

    # s = gen_rand_str()
    # m1.get_traces().to_csv('./../model_res/traces/GR_8k_s_vt_poutlier05' + s + '.csv')
    # pd.Series(m1.dic).to_csv('./../model_res/dic/GR_8k_s_vt_poutlier05' + s + 'dic.csv')

    
    # Add trial-wise  
    print("**New run 2**")
    m1wtw_no_sz = hddm.HDDMRegressor(
                sdf,
                vs,
                group_only_regressors=False,
                keep_regressor_trace=True, 
                p_outlier=0.05,
                include=['z', 'st', 'sv'],
                group_only_nodes=['sv', 'st', 'sz'], 
                )

    m1wtw_no_sz.find_starting_values()
    m1wtw_no_sz.sample(n_samples, n_burns)

    s = gen_rand_str()
    m1wtw_no_sz.get_traces().to_csv('./../model_res/traces/GR_run_also8k_ddm_add_trialwise_NO-SZ_s_vt_poutlier05' + s + '.csv')
    pd.Series(m1wtw_no_sz.dic).to_csv('./../model_res/dic/GR_run_also8k_ddm_add_trialwise_NO-SZ_s_vt_poutlier05' + s + 'dic.csv')

if split_half == 1:
    # Reruns with p = .05  
    if odd == 0:    
        print("Even split half")
        
        tr0 = hddm.HDDMRegressor(even_sdf,
                                        {'model': 'v ~ 1 + val_ctr', 'link_func': lambda x: x}, 
                                        group_only_regressors=False,
                                        keep_regressor_trace=True, 
                                        p_outlier=0.05,
                                        include=['z', 'sv', 'st'],
                                        group_only_nodes=['sv', 'st', 'sz'], 
                    )

        tr0.sample(n_samples, n_burns)

        tr0.get_traces().to_csv("../model_res/traces/DDM_split_half_even_wtrialwise_poutlier05" + s + ".csv")
    
    if odd == 1:
        print("Odd split half")
        
        tr1 = hddm.HDDMRegressor(odd_sdf,
                                        {'model': 'v ~ 1 + val_ctr', 'link_func': lambda x: x}, 
                                        group_only_regressors=False,
                                        keep_regressor_trace=True, 
                                        p_outlier=0.05,
                                        include=['z', 'sv', 'st'],
                                        group_only_nodes=['sv', 'st', 'sz'], 
                                        )

        tr1.sample(n_samples, n_burns)
        tr1.get_traces().to_csv("../model_res/traces/DDM_split_half_odd__wtrialwise_poutlier05" + s + ".csv")

    # if which_model == 0:    
    #     print("Even")
    #     print(pre_sdf)
    #     tr0 = hddm.HDDMRegressor(even_sdf,
    #                                     {'model': 'v ~ 1 + val_ctr', 'link_func': lambda x: x}, 
    #                                     group_only_regressors=False,
    #                                     keep_regressor_trace=True, 
    #                                     p_outlier=0,
    #                                     include=['z'],
    #                                     group_only_nodes=['sv', 'st', 'sz'], 
    #                 )

    #     tr0.sample(n_samples, n_burns)

    #     tr0.get_traces().to_csv("../model_res/traces/DDM_split_half_even" + s + ".csv")
    
    # if which_model == 1:
    #     print("Odd")
    #     print(post_sdf)
    #     tr1 = hddm.HDDMRegressor(odd_sdf,
    #                                     {'model': 'v ~ 1 + val_ctr', 'link_func': lambda x: x}, 
    #                                     group_only_regressors=False,
    #                                     keep_regressor_trace=True, 
    #                                     p_outlier=0,
    #                                     include=['z'],
    #                                     group_only_nodes=['sv', 'st', 'sz'], 
    #                                     )

    #     tr1.sample(n_samples, n_burns)
    #     tr1.get_traces().to_csv("../model_res/traces/DDM_split_half_odd" + s + ".csv")

if test_retest == 1:

    if pre == 1:    
        print("Pre")
        print(pre_sdf)
        tr0 = hddm.HDDMRegressor(pre_sdf,
                                        {'model': 'v ~ 1 + val_ctr', 'link_func': lambda x: x}, 
                                        group_only_regressors=False,
                                        keep_regressor_trace=True, 
                                        p_outlier=0.05,
                                        include=['z', 'sv', 'st'],
                                        group_only_nodes=['sv', 'st', 'sz'], 
                    )

        tr0.sample(n_samples, n_burns)

        tr0.get_traces().to_csv("../model_res/traces/DDM_test_retest_PRE_s-val_wtrialwise_poutlier05" + s + ".csv")

    
    if pre == 0:
        print("Post")
        print(post_sdf)
        tr1 = hddm.HDDMRegressor(post_sdf,
                                        {'model': 'v ~ 1 + val_ctr', 'link_func': lambda x: x}, 
                                        group_only_regressors=False,
                                        keep_regressor_trace=True, 
                                        p_outlier=0.05,
                                        include=['z', 'sv', 'st'],
                                        group_only_nodes=['sv', 'st', 'sz'], 
                                        )

        tr1.sample(n_samples, n_burns)

        tr1.get_traces().to_csv("../model_res/traces/DDM_test_retest_POST_s-val_wtrialwise_poutlier05" + s + ".csv")
        #pd.Series(tr1.dic).to_csv("../model_res/dic/DDM_test_retest_POST_hdnn_sret_m0_sret_v_val_dic" + s + ".csv")


