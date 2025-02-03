#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:33:12 2024

@author: annabel
"""
from collections import OrderedDict


def concat_dicts(dict_lst):
    out = OrderedDict()
    for d in dict_lst:
        out = OrderedDict({**out, **d})
    return out


###############################################################################
### dictionaries from evoparam_blocks   #######################################
###############################################################################
def get_evoparam_config(block_name):
    if block_name == 'Placeholder':
        return OrderedDict({})
    
    if block_name == 'SelectMaskNorm':
        return OrderedDict({"use_anc_emb": "[BOOL]",
                            "use_desc_emb": "[BOOL]",
                            "norm_type": "[ STR=null: ('layer', 'RMS', null) ]"})
    
    
    elif block_name == 'FeedforwardToEvoparams':
        return OrderedDict({"layer_sizes": "[list of INT]",
                            "use_anc_emb": "[BOOL]",
                            "use_desc_emb": "[BOOL]",
                            "act_type": "[ STR='relu': ('relu','silu','gelu')",
                            "dropout": "[FLOAT=0.0]",
                            "norm_type": "[ STR=null: ('layer', 'RMS', null) ]"})
    
    
    elif block_name == 'EvoparamsFromFile':
        return OrderedDict({"load_from_file": "[STR]"})
    
    
    elif block_name == 'OneExchangeabilityMat':
        return OrderedDict({"emission_alphabet_size": "[INT]",
                            "manual_init": "[BOOL]",
                            "load_from_file": "[STR; use this option if manual_init == True]"})
    
    
    elif block_name == 'OneParamVec':
        return OrderedDict({"num_evoparams": "[INT]",
                            "evoparams_name": "[STR]",
                            "manual_init": "[BOOL]",
                            "load_from_file": "[STR; use this option if manual_init == True]"})
    
    
    elif block_name == 'ExchangeabilityMatFromEmbeds':
        return OrderedDict({"emission_alphabet_size": "[INT]",
                            "use_bias": "[BOOL=false]"})
    
    
    elif block_name == 'ParamVecFromEmbeds':
        return OrderedDict({"num_evoparams": "[INT]",
                            "evoparams_name": "[STR]",
                            "use_bias": "[BOOL=false]"})


###############################################################################
### dictionaries from pairHMM_emissions_blocks   ##############################
###############################################################################
def get_emission_block_config(block_name):
    if block_name == 'MatchEmissionsLogprobs':
        return OrderedDict({"unit_norm_rate_matrix": "[BOOL]"})
    
    elif block_name == 'MatchEmissionsLogprobsFromFile':
        return OrderedDict({"load_from_file": "[STR]"})
    
    elif block_name == 'InsEmissionsLogprobs':
        return OrderedDict({})
    
    elif block_name == 'InsEmissionsLogprobsFromFile':
        return OrderedDict({"load_from_file": "[STR]"})


###############################################################################
### dictionaries from pairHMM_transitions_blocks   ############################
###############################################################################
def get_transition_block_config(block_name):
    if block_name == 'NoIndels':
        return OrderedDict({})
    
    elif block_name == 'TKF91TransitionLogprobs':
        return OrderedDict({"tie_weights": "[BOOL]",
                            "safe_grads": "[BOOL]",
                            "tkf_err": "[FLOAT=1e-4",
                            "eos_as_match": "[BOOL]"})
    
    elif block_name == 'TKF92TransitionLogprobs':
        return OrderedDict({"tie_weights": "[BOOL]",
                            "safe_grads": "[BOOL]",
                            "tkf_err": "[FLOAT=1e-4]",
                            "eos_as_match": "[BOOL]"})
    
    elif block_name == 'TransitionLogprobsFromFile':
        return OrderedDict({"load_from_file": "[STR]"})
    


###############################################################################
### MAIN   ####################################################################
###############################################################################
def create_config(process_embeds_for_exchang_module: str,
                  exchang_module: str,
                  
                  process_embeds_for_equilibr_module: str,
                  equilibr_module: str,
                  
                  process_embeds_for_indels_module: str,
                  indels_module: str,

                  emit_match_logprobs_module: str,
                  emit_ins_logprobs_module: str,
                  transits_logprobs_module: str):
    
    exchang_config = concat_dicts([get_evoparam_config(exchang_module),
                                   get_evoparam_config(process_embeds_for_exchang_module),
                                   get_emission_block_config(emit_match_logprobs_module)]
                                  )
    
    equilibr_config = concat_dicts([get_evoparam_config(equilibr_module),
                                   get_evoparam_config(process_embeds_for_equilibr_module),
                                   get_emission_block_config(emit_ins_logprobs_module)]
                                  )
    
    indels_config = concat_dicts([get_evoparam_config(indels_module),
                                   get_evoparam_config(process_embeds_for_indels_module),
                                   get_transition_block_config(transits_logprobs_module)]
                                  )
    
    
    return OrderedDict( {"process_embeds_for_exchang_module": process_embeds_for_exchang_module,
                         "exchang_module": exchang_module,
                         
                         "LINEBREAK1": "",
                         
                         "process_embeds_for_equilibr_module": process_embeds_for_equilibr_module,
                         "equilibr_module": equilibr_module,
                         
                         "LINEBREAK2": "",
                         
                         "process_embeds_for_indels_module": process_embeds_for_indels_module,
                         "indels_module": indels_module,
                         
                         "LINEBREAK301": "",
                         
                         "emit_match_logprobs_module": emit_match_logprobs_module,
                         "emit_ins_logprobs_module": emit_ins_logprobs_module,
                         "transits_logprobs_module": transits_logprobs_module,
                         
                         "LINEBREAK3": "",
                         
                         "emission_alphabet_size": "[INT]",
                         "norm_loss_by": "[ STR = 'desc_len': ('desc_len', 'align_len') ]",
                         "times_from": "[ STR: ('geometric', 't_array_from_file', 'one_time_per_sample') ]",
                         
                         "LINEBREAK401": "",
                         
                         "t_grid_center": "[FLOAT; use if times_from = geometric]",
                         "t_grid_step": "[FLOAT; use if times_from = geometric]",
                         "t_grid_num_steps": "[INT; use if times_from = geometric]",
                         
                         "LINEBREAK402": "",
                         
                         "times_file": "[STR]; use if times_from = t_array_from_file",
                         "const_for_time_marg": "[FLOAT]; use if times_from = t_array_from_file",
                         
                         "LINEBREAK4": "",
                         
                         "exchang_config": exchang_config,
                         
                         "LINEBREAK5": "",
                         
                         "equilibr_config": equilibr_config,
                         
                         "LINEBREAK6": "",
                         
                         "indels_config": indels_config
                         } )
