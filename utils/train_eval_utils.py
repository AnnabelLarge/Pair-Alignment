#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 18:06:40 2025

@author: annabel

These are used in train and/or eval scripts
"""
import jax
import jax.numpy as jnp


###############################################################################
### BOTH TRAIN AND EVAL   #####################################################
###############################################################################
def selective_squeeze(mat):
    """
    jnp.squeeze, but ignore batch dimension (dim0)
    """
    new_shape = tuple( [mat.shape[0]] + [s for s in mat.shape[1:] if s != 1] )
    return jnp.reshape(mat, new_shape)

def clip_by_bins(datamat, 
                 chunk_length: int = 512, 
                 padding_idx = 0):
    """
    Clip excess paddings by binning according to chunk_length
    
    For example, if chunk_length is 3, then possible places to clip include:
        > up to length 3, if longest sequence is <= 3 in length
        > up to length 6, if longest sequence is > 3 and <= 6 in length
        > up to length 9, if longest sequence is > 6 and <= 9 in length
        > etc., until maximum length of batch_seqs
    
    overall, this helps jit-compile different versions of the functions
      for different max lengths (semi-dynamic batching)
     
        
    Arguments:
    ----------
    datamat : ArrayLike
        dim 1 MUST be a length dim!!!
    
    chunk_length : int = 512
        length of the chunk
    
    padding_idx : int = 0
        padding token
    """
    # lengths
    L_max = datamat.shape[1]
    max_len_without_padding = (datamat != padding_idx).sum(axis=1).max()
    
    # determine the number of chunks
    def cond_fun(num_chunks):
        return chunk_length * num_chunks < max_len_without_padding

    def body_fun(num_chunks):
        return num_chunks + 1
    
    num_chunks = jax.lax.while_loop(cond_fun, body_fun, 1)
    length_with_all_chunks = chunk_length * num_chunks
    
    # if length_with_all_chunks is greater than max_len, 
    #   use max_len instead
    clip_to = jnp.where( length_with_all_chunks > max_len,
                         L_max,
                         length_with_all_chunks )
    return clip_to


def determine_seqlen_bin(batch,
                         chunk_length: int,
                         seq_padding_idx: int = 0):
    unaligned_seqs = batch[0]
    batch_max_seqlen = clip_by_bins(batch_seqs = unaligned_seqs, 
                                    chunk_length = chunk_length, 
                                    padding_idx = seq_padding_idx)
    return batch_max_seqlen


def determine_alignlen_bin(batch,
                           chunk_length: int,
                           seq_padding_idx: int = 0):
    ### batch has 4 entries:
    ### 0.) unaligned seqs: (B, L, 2)
    ### 1.) aligned matrices: (B, L, 2)
    ### 2.) time (optional): (B,) or None
    ### 3.) dataloader idx (B,)
    # use the first sequence from aligned matrix for this (gapped ancestor for 
    #   neural_pairhmm, alignment-augmented descendant for feedforward); 
    #   exclude <bos> for the clip_by_bins function
    gapped_seq = batch[1][:, 1:, 0]
    
    # get length
    batch_max_alignlen = clip_by_bins(batch_seqs = gapped_seq, 
                                      chunk_length = chunk_length, 
                                      padding_idx = seq_padding_idx)
      
    # add one again, to re-include <bos>
    return batch_max_alignlen + 1

def write_approx_dict(approx_dict, 
                      out_arrs_dir,
                      out_file,
                      subline,
                      calc_sum = True):
    """
    record if you used TKF approximation functions
    """
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



###############################################################################
### TRAIN ONLY   ##############################################################
###############################################################################
def write_timing_file(outdir,
                       train_times,
                       eval_times,
                       total_times):
    """
    record real and cpu times during training
    """
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


def setup_training_dir(args):
    if 'assert_no_overwrite' not in dir(args):
        args.assert_no_overwrite = True
    
    ### create folder/file names
    tboard_dir = f'{os.getcwd()}/{args.training_wkdir}/tboard/{args.training_wkdir}'
    model_ckpts_dir = f'{os.getcwd()}/{args.training_wkdir}/model_ckpts'
    logfile_dir = f'{os.getcwd()}/{args.training_wkdir}/logfiles'
    out_arrs_dir = f'{os.getcwd()}/{args.training_wkdir}/out_arrs'
    
    # create logfile in the logfile_dir
    logfile_filename = f'PROGRESS.log'
    
    
    ### what to do if training directory exists
    # OPTION 1: IF TRAINING WKDIR ALREAD EXISTS, RAISE RUN TIME ERROR
    if os.path.exists(f'{os.getcwd()}/{args.training_wkdir}') and args.assert_no_overwrite:
        raise RuntimeError(f'{args.training_wkdir} ALREADY EXISTS; DOES IT HAVE DATA?')
    
    # # OPTION 2: IF TRAINING WKDIR ALREADY EXISTS, DELETE IT 
    # elif os.path.exists(f'{os.getcwd()}/{args.training_wkdir}') and not args.assert_no_overwrite:
    #     shutil.rmtree(f'{os.getcwd()}/{args.training_wkdir}')
    
    
    ### make training wkdir and subdirectories
    if not os.path.exists(f'{os.getcwd()}/{args.training_wkdir}'):
        os.mkdir(f'{os.getcwd()}/{args.training_wkdir}')
        os.mkdir(model_ckpts_dir)
        os.mkdir(logfile_dir)
        os.mkdir(out_arrs_dir)
        # tensorboard directory takes care of itself
    
    
    ### add these filenames to the args dictionary, to be passed to training
    ### script
    args.tboard_dir = tboard_dir
    args.model_ckpts_dir = model_ckpts_dir
    args.logfile_dir = logfile_dir
    args.logfile_name = f'{logfile_dir}/{logfile_filename}'
    args.out_arrs_dir = out_arrs_dir
        
        

###############################################################################
### EVAL ONLY   ###############################################################
###############################################################################

