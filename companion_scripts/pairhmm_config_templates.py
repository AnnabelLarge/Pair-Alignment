#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 15:46:26 2024

@author: annabel

"""
import json
from collections import OrderedDict


def dict_to_json(d: dict):
    json_out_raw = json.dumps(d, indent = 4).split('\n')
    
    json_out = ''
    for line in json_out_raw:
        if 'LINEBREAK' in line:
            json_out += '\n'
        else:
            json_out += line + '\n'
    
    return json_out

def concat_dicts(dict_lst):
    out = OrderedDict()
    for d in dict_lst:
        out = OrderedDict({**out, **d})
    return out



###############################################################################
### Main   ####################################################################
###############################################################################
def make_pairhmm_train_config(load_all: bool):
    from models.simple_site_class_predict.create_config import create_config
    
    pred_config = create_config(load_all = load_all)
    
    
    ### rest of the config file
    out = OrderedDict({"training_wkdir": "[STR]",
                       "rng_seednum": "[INT]",
                       
                       "LINEBREAK100":"",
                       
                       "data_dir": "[STR]",
                       "train_dset_splits": "[list of STR]",
                       "test_dset_splits": "[list of STR]",
                       "toss_alignments_longer_than": "[INT]",
                       "(if markovian sites) chunk_length": "[INT]",
                       "(if independent sites) bos_eos_as_match": "[BOOL]",
                       "batch_size": "[INT]",
                       
                       "LINEBREAK101":"",
                       
                       "norm_loss_by": "[STR='desc_len', 'align_len']",
                       "(if not feedforward) loss_type": "[STR='joint','cond']",
                       
                       "LINEBREAK102":"",
                       
                       "num_epochs": "",
                       "optimizer_config": OrderedDict({
                           "init_value": "[FLOAT]",
                           "peak_value": "[FLOAT]",
                           "end_value": "[FLOAT]",
                           "warmup_steps": "[INT]",
                           "weight_decay": "[FLOAT]",
                           "every_k_schedule": "[INT]"
                           }),
                       "early_stop_cond1_atol": "[FLOAT]",
                       "early_stop_cond2_gap": "[FLOAT]",
                       "patience": "[INT]",
                       
                       "LINEBREAK103":"",
                       
                       "use_scan_fns": "[BOOL]",
                       "chunk_length": "[INT]",
                       "toss_alignments_longer_than": "[INT, None]",
                       
                       "LINEBREAK203": "",
                       
                       "interms_for_tboard": OrderedDict({
                           "finalpred_sow_outputs":"[BOOL]",
                           "forward_pass_outputs":"[BOOL]",
                           }),
                       "save_arrs": "[BOOL]",
                       "histogram_output_freq": "[INT]",
                       
                       "LINEBREAK104":"",
                       
                       "pred_model_type": "[STR = pairhmm_indp_sites, pairhmm_markovian_sites]",
                       'pred_config': pred_config})
    
    json_out = dict_to_json(out)
    return json_out


if __name__ == "__main__":
    to_write = make_pairhmm_train_config(load_all = False)
    
    with open(f'pairHMM_train_template.json','w') as g:
        g.write(to_write)
    
