#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 12:28:28 2025

@author: annabel
"""
import pandas as pd
import os
import sys
import pickle
from tqdm import tqdm


def main(fold):
    all_dirs = [d for d in os.listdir(fold) if d.startswith('RESULTS')]
    
    all_out = []
    for d in tqdm(all_dirs):
        name_in_parts = d.split('_')
        assert len(name_in_parts) == 5

        times_used = name_in_parts[1]
        trial_name = name_in_parts[3]
        rand_seed = name_in_parts[4]
        out = {'times_used': times_used,
               'trial_name': trial_name,
               'rand_seed': rand_seed}

        del name_in_parts

        path = f'{fold}/{d}/logfiles/AVE-LOSSES.tsv'
        to_add = pd.read_csv(path, sep='\t', header=None, index_col=0).to_dict()[1]

        out = {**out, **to_add}
        all_out.append(out)

    all_out = pd.DataFrame(all_out)
    all_out = all_out[['RUN',
                       'times_used',
                       'trial_name',
                       'rand_seed',
                       'train_ave_loss',
                       'train_ave_loss_seqlen_normed',
                       'train_perplexity',
                       'train_ece',
                       'test_ave_loss',
                       'test_ave_loss_seqlen_normed',
                       'test_perplexity',
                       'test_ece']]
    all_out = all_out.sort_values(by='trial_name')
    
    idx = all_out.groupby('trial_name')['test_ece'].idxmin()
    best = all_out.loc[idx]
    
    # save
    all_out.to_csv(f'{fold}/ALL_test_ece.tsv', sep='\t')
    best.to_csv(f'{fold}/BEST_test_ece.tsv', sep='\t')
    

if __name__ == '__main__':
    import sys
    
    fold = sys.argv[1]    
    main(fold)
