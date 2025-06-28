#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 18:23:14 2025

@author: annabel
"""

def write_approx_dict(approx_dict, 
                      out_arrs_dir,
                      out_file,
                      subline,
                      calc_sum = True):
    used_approx = False
    to_write = ''
    
    key_lst = [key for key in approx_dict.keys() if key != 't_array']
    for key in key_lst:
        val = approx_dict[key]
        if val.any():
            used_approx = True
            if calc_sum:
                approx_count = val.sum()
                to_write += f'{key}: {approx_count}\n'
            else:
                to_write += f'{key}: {val}\n'
            
    if used_approx:
        
        # for pairHMMs, also record time
        if 't_array' in approx_dict.keys():
            t_to_write = approx_dict['t_array']
            t_to_write = t_to_write[t_to_write != -1.]
            t_to_write = ', '.join( list(set([str(t) for t in t_to_write])) )
            with open(f'{out_arrs_dir}/{out_file}','a') as g:
                g.write(f'{subline}\n')
                g.write(f'times: {t_to_write}\n')
                g.write(f'({len(t_to_write)} times)\n\n')
                
        # for neural TKF, only have sums
        with open(f'{out_arrs_dir}/{out_file}','a') as g:
            g.write(f'{subline}\n')
            g.write(to_write + '\n')
            g.write('\n')
        
    del used_approx, to_write, key, val
    
