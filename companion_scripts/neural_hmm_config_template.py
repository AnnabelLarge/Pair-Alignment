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


###############################################################################
### Used when building a training config   ####################################
###############################################################################
def get_seq_model_config(model_type: str, 
                         extra_args: dict = dict()):
    if model_type in ['OneHot', 'Masking', None]:
        from models.sequence_embedders.no_params.create_config import create_config
        
    elif model_type == 'CNN':
        from models.sequence_embedders.cnn.create_config import create_config
        
    elif model_type == 'LSTM':
        from models.sequence_embedders.lstm.create_config import create_config
        
    elif model_type == 'Transformer':
        from models.sequence_embedders.transformer.create_config import create_config
        
    elif model_type == 'Mamba':
        from models.sequence_embedders.mamba.create_config import create_config
    
    return create_config(extra_args)


def get_pred_head_config(model_type: str, 
                         extra_args: dict = dict()):
    if model_type == 'feedforward':
        from models.feedforward_predict.create_config import create_config
    
    elif model_type == 'neural_hmm':
        from models.neural_hmm_predict.create_config import create_config
    
    return create_config(**extra_args)



###############################################################################
### Main   ####################################################################
###############################################################################
def make_train_config(anc_model_type: str,
                      desc_model_type: str,
                      pred_model_type: str,
                      pred_model_kwargs: dict = dict()
                      ):
    ### ancestor model
    if anc_model_type in ['Mamba']:
        anc_model_kwargs = {'bidirect': True}
    
    else:
        anc_model_kwargs = dict()
        
    anc_enc_config = get_seq_model_config(model_type = anc_model_type,
                                          extra_args = anc_model_kwargs)
    
    
    ### descendant model
    if desc_model_type in ['Mamba']:
        desc_model_kwargs = {'bidirect': False}
    
    else:
        desc_model_kwargs = dict()
        
    desc_dec_config = get_seq_model_config(model_type = desc_model_type,
                                           extra_args = desc_model_kwargs)
    
    
    ### prediction head
    # make config
    pred_config = get_pred_head_config(model_type = pred_model_type, 
                                       extra_args = pred_model_kwargs)
    
    
    ### rest of the config file
    out = OrderedDict({"training_wkdir": "[STR]",
                       "rng_seednum": "[INT]",
                       
                       "LINEBREAK100":"",
                       
                       "data_dir": "[STR]",
                       "train_dset_splits": "[list of STR]",
                       "test_dset_splits": "[list of STR]",
                       "toss_alignments_longer_than": "[INT]",
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
                           "decoder_sow_outputs":"[BOOL]",
                           "encoder_sow_outputs":"[BOOL]",
                           "finalpred_sow_outputs":"[BOOL]",
                           "gradients":"[BOOL]",
                           "weights":"[BOOL]",
                           "optimizer": "[BOOL]",
                           "ancestor_embeddings":"[BOOL]",
                           "descendant_embeddings":"[BOOL]",
                           "forward_pass_outputs":"[BOOL]",
                           "final_logprobs":"[BOOL]"
                           }),
                       "save_arrs": "[BOOL]",
                       "histogram_output_freq": "[INT]",
                       
                       "LINEBREAK104":"",
                       
                       "anc_model_type": anc_model_type,
                       "anc_enc_config": anc_enc_config,
                       
                       "LINEBREAK105":"",
                       
                       "desc_model_type": desc_model_type,
                       "desc_dec_config": desc_dec_config,
                       
                       "LINEBREAK106":"",
                       
                       "pred_model_type": pred_model_type,
                       'pred_config': pred_config})
    
    json_out = dict_to_json(out)
    return json_out


if __name__ == '__main__':
    # initializers = { 'base_hmm_fit_indel_params': base_hmm_fit_indel_params,
    #                  'base_hmm_load_all': base_hmm_load_all,
    #                  'local_exch_equilibr': local_exch_equilibr,
    #                  'local_exch_equilibr_r': local_exch_equilibr_r,
    #                  'all_local': all_local }
    preset_name = 'base_hmm_load_all'
    to_write = make_train_config( anc_model_type= None,
                                  desc_model_type= None,
                                  pred_model_type= 'neural_hmm',
                                  pred_model_kwargs= {'preset_name': preset_name}
                                  )
    
    with open(f'{preset_name}_template.json','w') as g:
        g.write(to_write)


# def make_eval_config():
#     out = OrderedDict({"eval_wkdir": "[STR]",
#                        "rng_seednum": "[INT]",
#                        "training_wkdir": "[STR]",
                       
#                        "LINEBREAK200":"",

#                        "data_dir": "[STR]",
#                        "test_dset_splits": "[list of STR]",
#                        "batch_size": "[INT]",
                       
#                        "LINEBREAK103":"",
                       
#                        "use_scan_fns": "[BOOL]",
#                        "chunk_length": "[INT]",
#                        "toss_alignments_longer_than": "[INT, None]",

#                        "LINEBREAK201":"",
                       
#                        "interms_for_tboard": OrderedDict({
#                            "decoder_sow_outputs":"[BOOL]",
#                            "encoder_sow_outputs":"[BOOL]",
#                            "finalpred_sow_outputs":"[BOOL]",
#                            "gradients":"[BOOL]",
#                            "weights":"[BOOL]",
#                            "optimizer": "[BOOL]",
#                            "ancestor_embeddings":"[BOOL]",
#                            "descendant_embeddings":"[BOOL]",
#                            "forward_pass_outputs":"[BOOL]",
#                            "final_logprobs":"[BOOL]"
#                            }),
#                        "save_arrs": "[BOOL]",
#                        "output_attn_weights": "[BOOL]"
#                        })
    
#     json_out = dict_to_json(out)
#     return json_out

    