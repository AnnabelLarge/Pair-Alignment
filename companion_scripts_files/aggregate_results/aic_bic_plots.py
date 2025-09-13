#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 14:12:03 2025

@author: annabel
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib import rc
plt.rcParams["font.family"] = 'Optima'
plt.rcParams["font.size"] = 18
plt.rc('axes', unicode_minus=False)


n_samples_in_train = 848834
file = 'BEST_TIME_PARAM-COUNTS_pairhmms.tsv'

def aic(k, loglike):
    """
    k: number of parameters
    loglike: sum of joint loglikelihood on training set
    """
    if not loglike < 0:
        loglike = -loglike
    return 2 * k - 2 * loglike

def bic(k, n, loglike):
    """
    k: number of parameters
    n: number of training samples
    loglike: sum of joint loglikelihood on training set
    """
    if not loglike < 0:
        loglike = -loglike
    return k * np.log(n) - 2 * loglike

df = pd.read_csv(file, sep='\t', index_col=0)

### add aic, bic
aic_col = {}
bic_col = {}
for i in range(len(df)):
    row = df.iloc[i]
    run = row['RUN']
    k = row['num_parameters'].item()
    loglike = row['train_sum_joint_loglikes'].item()
    
    aic_col[run] = aic(k = k,
                       loglike = loglike)
    
    bic_col[run] = bic(k = k,
                       n = n_samples_in_train,
                       loglike = loglike)

df["aic"] = df["RUN"].map(aic_col)
df["bic"] = df["RUN"].map(bic_col)


### make plots by substitution model
def make_plots(subs_model, colname):
    if colname == 'num_parameters':
        xax_lab = 'Number of parameters'
    
    elif colname == 'num_site_mixtures':
        xax_lab = 'Number of mixtures'
    
    aic_fig, aic_ax = plt.subplots(figsize=[10,8])
    bic_fig, bic_ax = plt.subplots(figsize=[10,8])
    
    def add_to_plots(subs_model,
                     mixture_type,
                     color):
        sub_df = df[ (df['subst_model_type'] == subs_model) &
                     (df['indel_model_type'] != 'tkf91') &
                     (df['type'] == mixture_type)]
        sub_df = sub_df[[colname,
                         'aic',
                         'bic']]
        sub_df = sub_df.sort_values(by=colname)
        
        label = mixture_type.replace(' pairhmm','')
        
        aic_ax.plot( sub_df[colname],
                     sub_df['aic'],
                     '-o',
                     color = color,
                     label = label)
        
        bic_ax.plot( sub_df[colname],
                     sub_df['bic'],
                     '-o',
                     color = color,
                     label = label)
        
        del sub_df, label
    
    add_to_plots(subs_model = subs_model,
                 mixture_type = 'reference pairhmm',
                 color = 'tab:blue')
    
    add_to_plots(subs_model = subs_model,
                 mixture_type = 'site mix pairhmm',
                 color = 'tab:orange')
    
    add_to_plots(subs_model = subs_model,
                 mixture_type = 'fragment mix pairhmm',
                 color = 'tab:green')
    
    add_to_plots(subs_model = subs_model,
                 mixture_type = 'domain mix pairhmm',
                 color = 'tab:purple')
    
    aic_ax.grid()
    aic_ax.legend()
    aic_ax.set_xlabel(xax_lab)
    aic_ax.set_ylabel('AIC')
    aic_ax.set_title(f'AIC of {subs_model} mixture models')
    
    bic_ax.grid()
    bic_ax.legend()
    bic_ax.set_xlabel(xax_lab)
    bic_ax.set_ylabel('BIC')
    bic_ax.set_title(f'BIC of {subs_model} mixture models')
    
    aic_fig.savefig(f'AIC_{subs_model}_{colname}.png')
    bic_fig.savefig(f'BIC_{subs_model}_{colname}.png')


# make_plots(subs_model = 'f81', colname='num_parameters')
# make_plots(subs_model = 'f81', colname='num_site_mixtures')
# make_plots(subs_model = 'gtr', colname='num_parameters')
# make_plots(subs_model = 'gtr', colname='num_site_mixtures')



### combine f81 and gtr into one plot
def make_combined_plots(colname, zoom=None):
    if colname == 'num_parameters':
        xax_lab = 'Number of parameters'
    
    elif colname == 'num_site_mixtures':
        xax_lab = 'Number of mixtures'
    
    aic_fig, aic_ax = plt.subplots(figsize=[10,8])
    bic_fig, bic_ax = plt.subplots(figsize=[10,8])
    
    def add_to_plots(subs_model,
                     mixture_type,
                     color):
        sub_df = df[ (df['subst_model_type'] == subs_model) &
                     (df['indel_model_type'] != 'tkf91') &
                     (df['type'] == mixture_type)]
        sub_df = sub_df[[colname,
                         'aic',
                         'bic']]
        sub_df = sub_df.sort_values(by=colname)
        
        label = mixture_type.replace(' pairhmm',f': {subs_model}')
        
        aic_ax.plot( sub_df[colname],
                     sub_df['aic'],
                     '-o',
                     color = color,
                     label = label)
        
        bic_ax.plot( sub_df[colname],
                     sub_df['bic'],
                     '-o',
                     color = color,
                     label = label)
        
        del sub_df, label
    
    # f81
    add_to_plots(subs_model = 'f81',
                 mixture_type = 'reference pairhmm',
                 color = 'deepskyblue')
    
    add_to_plots(subs_model = 'f81',
                 mixture_type = 'site mix pairhmm',
                 color = 'goldenrod')
    
    add_to_plots(subs_model = 'f81',
                 mixture_type = 'fragment mix pairhmm',
                 color = 'mediumseagreen')
    
    add_to_plots(subs_model = 'f81',
                 mixture_type = 'domain mix pairhmm',
                 color = 'plum')
    
    # gtr
    add_to_plots(subs_model = 'gtr',
                 mixture_type = 'reference pairhmm',
                 color = 'darkblue')
    
    add_to_plots(subs_model = 'gtr',
                 mixture_type = 'site mix pairhmm',
                 color = 'peru')
    
    add_to_plots(subs_model = 'gtr',
                 mixture_type = 'fragment mix pairhmm',
                 color = 'darkgreen')
    
    add_to_plots(subs_model = 'gtr',
                 mixture_type = 'domain mix pairhmm',
                 color = 'darkviolet')
    
    aic_ax.grid()
    aic_ax.legend()
    aic_ax.set_xlabel(xax_lab)
    aic_ax.set_ylabel('AIC')
    
    bic_ax.grid()
    bic_ax.legend()
    bic_ax.set_xlabel(xax_lab)
    bic_ax.set_ylabel('BIC')
    
    if zoom:
        aic_ax.set_xlim(zoom)
        bic_ax.set_xlim(zoom)
        colname = colname + '_ZOOMED'
        title_suff = '(ZOOMED IN)'
    
    else:
        title_suff = ''
    
    aic_ax.set_title(f'AIC of mixture models {title_suff}')
    bic_ax.set_title(f'BIC of mixture models {title_suff}')
    
    aic_fig.savefig(f'AIC_all-models_{colname}.png')
    bic_fig.savefig(f'BIC_all-models_{colname}.png')

make_combined_plots(colname='num_parameters')
make_combined_plots(colname='num_parameters', zoom=[-10,1000])
make_combined_plots(colname='num_site_mixtures')