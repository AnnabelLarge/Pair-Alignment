#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 12:56:53 2025

@author: annabel
"""
def write_timing_file(outdir,
                       train_times,
                       eval_times,
                       total_times):
    num_nonzero_times = (total_times[...,0] > 0).sum(axis=0)
    
    if num_nonzero_times >= 1:
        first_epoch_train_time = train_times[0,:]
        first_epoch_eval_time = eval_times[0,:]
        first_epoch_total_time = total_times[0,:]
        
        with open(f'{outdir}/TIMING.txt','w') as g:
            g.write('# First epoch (with jit-compilation)\n')
            g.write(f'\t\treal\tcpu\n')
            
            g.write(f'train\t')
            g.write(f'{first_epoch_train_time[0].item()}\t')
            g.write(f'{first_epoch_train_time[1].item()}\n')
            
            g.write(f'eval\t')
            g.write(f'{first_epoch_eval_time[0].item()}\t')
            g.write(f'{first_epoch_eval_time[1].item()}\n')
            
            g.write(f'total\t')
            g.write(f'{first_epoch_total_time[0].item()}\t')
            g.write(f'{first_epoch_total_time[1].item()}\n')
            
            g.write(f'\n')
        
        if num_nonzero_times > 1:
            n = num_nonzero_times - 1
            
            following_train_times = train_times[1:,:].mean(axis=0)
            following_eval_times = eval_times[1:,:].mean(axis=0)
            following_total_times = total_times[1:,:].mean(axis=0)
            
            with open(f'{outdir}/TIMING.txt','a') as g:
                g.write(f'# Average over following {n} epochs\n')
                g.write(f'\t\treal\tcpu\n')
                
                g.write(f'train\t')
                g.write(f'{following_train_times[0].item()}\t')
                g.write(f'{following_train_times[1].item()}\n')
                
                g.write(f'eval\t')
                g.write(f'{following_eval_times[0].item()}\t')
                g.write(f'{following_eval_times[1].item()}\n')
                
                g.write(f'total\t')
                g.write(f'{following_total_times[0].item()}\t')
                g.write(f'{following_total_times[1].item()}\n')
    
    else:
        print('No times to record')
        
