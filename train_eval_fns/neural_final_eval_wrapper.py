#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 12:53:15 2024

@author: annabel

used for both feedforward and neural TKF92
"""
# regular python
import numpy as np
import pandas as pd
from tqdm import tqdm
import pgzip
import pickle
from typing import Optional

# jax
from jax import numpy as jnp

# custom imports
from utils.train_eval_utils import write_approx_dict
from utils.tensorboard_recording_utils import (calc_stats_during_final_eval,
                                               update_stats_dict,
                                               write_stats_to_tabular)


def final_eval_wrapper(dataloader, 
                       dataset, 
                       best_trainstates, 
                       jitted_determine_seqlen_bin,
                       jitted_determine_alignlen_bin,
                       eval_fn_jitted,
                       out_alph_size: Optional[int], 
                       save_arrs: bool,
                       save_per_sample_losses: bool,
                       interms_for_tboard: dict, 
                       logfile_dir: str,
                       out_arrs_dir: str, 
                       outfile_prefix: str, 
                       tboard_writer = None):
    ####################################
    ### WHICH INTERMEDIATES TO WRITE   #
    ####################################
    # booleans for determining which intermediate arrays to return
    return_anc_embs = interms_for_tboard['ancestor_embeddings']
    return_desc_embs = interms_for_tboard['descendant_embeddings']
    return_forward_pass_outputs = interms_for_tboard['forward_pass_outputs']
    
    # just in case: during final eval, won't have gradient info; overwrite
    interms_for_tboard['gradients'] = False
    
    
    #############################
    ### GO THROUGH DATALOADER   #
    #############################    
    ### final metrics to keep track of
    sum_cond_logprobs = 0
    final_ave_loss_seqlen_normed = 0
    final_perplexity = 0
    
    have_acc_metrics = (out_alph_size is not None)
    
    if have_acc_metrics:
        final_acc = 0
        final_confusion_matrix = jnp.zeros( (out_alph_size-1, out_alph_size-1) )
    
    if tboard_writer:
        final_stats_for_tboard = dict()
    
    for batch_idx, batch in tqdm( enumerate(dataloader), total=len(dataloader) ): 
        ##########################
        ### run model on a batch #
        ##########################
        ### unpack briefly to get max len and number of samples in the 
        ### batch; place in some bin (this controls how many jit 
        ### compilations you do)
        batch_max_seqlen = jitted_determine_seqlen_bin(batch = batch).item()
        batch_max_alignlen = jitted_determine_alignlen_bin(batch = batch).item()
            
        # eval
        eval_metrics = eval_fn_jitted(batch=batch, 
                                      all_trainstates=best_trainstates,
                                      max_seq_len=batch_max_seqlen,
                                      max_align_len=batch_max_alignlen)
        
        # check if any TKF approximations were used
        write_approx_dict_flag = eval_metrics.get('used_approx', None)
        if write_approx_dict_flag is not None:
            subline = f'batch {batch_idx}:'
            write_approx_dict( approx_dict = eval_metrics['used_approx'], 
                               out_arrs_dir = out_arrs_dir,
                               out_file = 'FINAL-EVAL_tkf_approx.tsv',
                               subline = subline,
                               calc_sum = True )
        del write_approx_dict_flag
         
        # always returned from eval_metrics:
        #     - loss; float
        #     - batch_ave_perpl; float
        #     - batch_ave_acc; float or None
        #     - sum_neg_logP; (B,)
        #     - neg_logP_length_normed; (B,)
        #     - perplexity_perSamp; (B,)
        
        # returned, if using feedforward prediction head:
        #     - acc_perSamp; (B,)
        #     - cm_perSamp; (B, out_alph_size-1, out_alph_size-1)
        
        # returned, if using neural TKF prediction head
        #     - used_approx
        
        # returned if flag active:
        #     - anc_layer_metrics
        #     - desc_layer_metrics
        #     - proj_layer_metrics
        #     - anc_attn_weights 
        #     - desc_attn_weights 
        #     - final_ancestor_embeddings
        #     - final_descendant_embeddings
        #     - any outputs from forward_pass_outputs
        
        
        #########################################
        ### start df; record metrics per sample #
        #########################################
        final_loglikes = dataset.retrieve_sample_names(batch[-1])
        
        final_loglikes['logP'] = eval_metrics['sum_neg_logP']
        final_loglikes['logP/normlength'] = eval_metrics['neg_logP_length_normed']
        final_loglikes['perplexity'] = eval_metrics['perplexity_perSamp']
        final_loglikes['dataloader_idx'] = batch[-1]
        
        num_samples_in_batch = eval_metrics['sum_neg_logP'].shape[0]
        
        # record mean values to buckets
        wf = ( num_samples_in_batch / len(dataset) )
        sum_cond_logprobs += final_loglikes['logP'].sum()
        final_ave_loss_seqlen_normed += final_loglikes['logP/normlength'].mean() * wf
        final_perplexity += eval_metrics['batch_ave_perpl'] * wf

        # model may or may not record accuracy as well    
        if have_acc_metrics:
            acc_perSamp = eval_metrics.get('acc_perSamp', None)
            final_loglikes['generative_accuracy'] = acc_perSamp
            final_acc += eval_metrics['batch_ave_acc'] * wf
        
        # write losses
        if save_per_sample_losses:
            # as dataframe
            final_loglikes.to_csv((f'{logfile_dir}/{outfile_prefix}_pt{batch_idx}_'+
                                  'FINAL-LOGLIKES.tsv'), sep='\t')
            
            # # as numpy array
            # # col1 is sample_idx, col2 is sum of the negative log-likleihoods
            # col1 = batch[-1]
            # col2 = eval_metrics['sum_neg_logP']
            # to_write = np.stack([col1, col2], axis=1)
            # with open(f'{logfile_dir}/NP-MAT_{outfile_prefix}_pt{batch_idx}_FINAL-LOGLIKES.npy', 'wb') as g:
            #     np.save(g, to_write)
            # del col1, col2, g, to_write
        
        
        ###############################################
        ### calculate stats for the batch as desired; #
        ###   output these to the same folder         #
        ###############################################
        stats_to_record = calc_stats_during_final_eval(all_trainstates= best_trainstates,
                                                       dict_of_values = eval_metrics, 
                                                       interms_for_tboard = interms_for_tboard,
                                                       top_level_tag = outfile_prefix)
        if len(stats_to_record) > 0:
            out_file = ( f'{out_arrs_dir}/'+
                         f'{outfile_prefix}_pt{batch_idx}_BATCH-STATS.tsv.gz' )
            with pgzip.open(out_file, 'wt') as g:
                for key, val in stats_to_record.items():
                    g.write(f'{key}\t{val}\n')
        
        # add them to collection at end for tensorboard
        if tboard_writer:
            final_stats_for_tboard = update_stats_dict(old_dict=final_stats_for_tboard,
                                                       new_dict=stats_to_record)
        
        
        #########################################################
        ### confusion matrix across all samples (if applicable) #
        #########################################################
        if have_acc_metrics:
            cm_perSamp = eval_metrics['cm_perSamp']
            cm_across_batch = cm_perSamp.sum(axis=0)
            final_confusion_matrix = final_confusion_matrix + cm_across_batch
        
        
        ############################
        ### other arrays to output #
        ############################
        if save_arrs:
            out_file = f'{out_arrs_dir}/{outfile_prefix}_pt{batch_idx}_ARRS.pkl.gz'
            to_write = {}
            def add_to_out_dict(value_to_write, flag, file_suffix):
                if (flag) and (value_to_write is not None):
                    to_write[file_suffix] = value_to_write
            
            ### save confusion matrix
            if have_acc_metrics:
                add_to_out_dict(value_to_write = eval_metrics['cm_perSamp'],  
                                flag = True,
                                file_suffix = 'PER-SAMPLE-CONFUSION-MAT')
            
            
            ### may have other intermediates; these are controlled by flags
            add_to_out_dict(value_to_write = eval_metrics.get('final_ancestor_embeddings',None),  
                            flag = return_anc_embs,
                            file_suffix = 'ANC-SEQ-EMBEDDINGS')
            
            add_to_out_dict(value_to_write = eval_metrics.get('final_descendant_embeddings',None), 
                            flag = return_desc_embs,
                            file_suffix = 'DESC-SEQ-CAUSAL-EMBEDDINGS')
            
            if return_forward_pass_outputs:
                for key in eval_metrics.keys():
                    if key.startswith('scormat_'):
                        value_to_save = eval_metrics[key]
                        add_to_out_dict(value_to_write = value_to_save, 
                                        flag = return_forward_pass_outputs,
                                        file_suffix = key.replace('scormat_','').upper())
                        
            
            if 'anc_attn_weights' in eval_metrics.keys():
                add_to_out_dict(value_to_write = eval_metrics['anc_attn_weights'], 
                                flag = True,
                                file_suffix = f'ANC-SEQ-ATTN-WEIGHTS')
            
            if 'desc_attn_weights' in eval_metrics.keys():
                add_to_out_dict(value_to_write = eval_metrics['desc_attn_weights'], 
                                flag = True,
                                file_suffix = f'DESC-SEQ-CAUSAL-ATTN-WEIGHTS')
            
            ### finally, output a compressed dictionary of arrays with pgzip
            with pgzip.open(out_file, "wb") as g:
                pickle.dump(to_write, g)
            del to_write, out_file
        
    
    ######################
    ### POST EVAL LOOP   #
    ######################
    # record the column order for the numpy matrix written earlier
    #if save_per_sample_losses:
        #with open(f'{logfile_dir}/COLS-FOR-NP-MAT_{outfile_prefix}_pt{batch_idx}_FINAL-LOGLIKES.tsv', 'w') as g:
        #    g.write(f'dataloader_idx\n')
        #    g.write(f'sum_cond_loglikes\n')
            
    # extract whole-dataset performance
    final_ave_loss = sum_cond_logprobs / len(dataset)
    final_ece = jnp.exp( final_ave_loss_seqlen_normed )
    summary_stats = {'sum_cond_logprobs': sum_cond_logprobs,
                     'cond_ave_loss': final_ave_loss, 
                     'cond_ave_loss_seqlen_normed':final_ave_loss_seqlen_normed,
                     'cond_perplexity':final_perplexity,
                     'cond_ece':final_ece}
    
    if have_acc_metrics:
        summary_stats['acc'] = final_acc
    
        # write whole-dataset confusion matrix, separately
        out_file = f'{out_arrs_dir}/{outfile_prefix}_CONFUSION-MAT.npz'
        np.savez_compressed(out_file, 
                            final_confusion_matrix=final_confusion_matrix)
        del out_file, final_confusion_matrix
    
    # write summary stats collected from final_stats_for_tboard
    #   top_layer_name has already been provided
    if tboard_writer:
        write_stats_to_tabular(flat_dict = final_stats_for_tboard,
                               writer_obj = tboard_writer)
        
    return summary_stats
