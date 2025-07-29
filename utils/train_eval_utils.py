#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 18:06:40 2025

@author: annabel

These are used across the majority of train and/or eval scripts
"""
import os

import jax
import jax.numpy as jnp

from utils.tensorboard_recording_utils import write_times


###############################################################################
### BOTH TRAIN AND EVAL   #####################################################
###############################################################################  
class metrics_for_epoch:
    def  __init__(self,
                  have_acc,
                  epoch_idx):
        self.have_acc = have_acc
        self.epoch_idx = epoch_idx
        
        self.epoch_ave_loss = 0
        self.epoch_ave_perpl = 0
        
        if self.have_acc:
            self.epoch_ave_acc = 0
            
    def update_after_batch(self,
                           batch_weight,
                           batch_loss,
                           batch_perpl,
                           batch_acc = None):
        self.epoch_ave_loss += batch_loss * batch_weight
        self.epoch_ave_perpl += batch_perpl * batch_weight
        if self.have_acc and (batch_acc is not None):
            self.epoch_ave_acc += batch_acc * batch_weight
    
    def write_epoch_metrics_to_tensorboard(self,
                                           writer,
                                           tag):
        writer.add_scalar(tag = f'Loss/{tag}', 
                          scalar_value = self.epoch_ave_loss.item(), 
                          global_step = self.epoch_idx)
        
        writer.add_scalar(tag=f'Perplexity/{tag}',
                          scalar_value=self.epoch_ave_perpl.item(), 
                          global_step=self.epoch_idx)
        
        if self.have_acc:
            writer.add_scalar(tag=f'Accuracy/{tag}',
                              scalar_value=self.epoch_ave_acc.item(), 
                              global_step=self.epoch_idx)    
        
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
    ### batch has 4 entries:
    ### 0.) unaligned seqs: (B, L, 2)
    ### 1.) aligned matrices: (B, L, 2)
    ### 2.) time (optional): (B,) or None
    ### 3.) dataloader idx (B,)
    unaligned_seqs = batch[0]
    batch_max_seqlen = clip_by_bins(batch_seqs = unaligned_seqs, 
                                    chunk_length = chunk_length, 
                                    padding_idx = seq_padding_idx)
    return batch_max_seqlen

def jit_compile_determine_seqlen_bin(args):
    parted_determine_seqlen_bin = partial(determine_seqlen_bin,
                                          chunk_length = args.chunk_length, 
                                          seq_padding_idx = args.seq_padding_idx)
    jitted_determine_seqlen_bin = jax.jit(parted_determine_seqlen_bin)
    return jitted_determine_seqlen_bin

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
    aligned_mats_excluding_bos = batch[1][:, 1:, 0]
    
    # get length
    batch_max_alignlen = clip_by_bins(batch_seqs = aligned_mats_excluding_bos, 
                                      chunk_length = chunk_length, 
                                      padding_idx = seq_padding_idx)
      
    # add one again, to re-include <bos>
    return batch_max_alignlen + 1

def jit_compile_determine_alignlen_bin(args):
    parted_determine_alignlen_bin = partial(determine_alignlen_bin,  
                                            chunk_length = args.chunk_length,
                                            seq_padding_idx = args.seq_padding_idx)
    jitted_determine_alignlen_bin = jax.jit(parted_determine_alignlen_bin)
    return jitted_determine_alignlen_bin
    

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

def write_final_eval_results(args,
                             summary_stats: dict,
                             filename: str):
    to_write_prefix = {'RUN': args.training_wkdir}
    to_write = {**to_write_prefix, **summary_stats}
    
    with open(f'{args.logfile_dir}/{filename}','w') as g:
        for k, v in to_write.items():
            g.write(f'{k}\t{v}\n')


###############################################################################
### TRAIN ONLY   ##############################################################
###############################################################################
class best_models_base_class:
    def  __init__(self,
                  initial_trainstate_objs: list):
        self.best_epoch = -1
        self.best_test_loss = jnp.finfo(jnp.float32).max
        self.best_trainstates = initial_trainstate_objs
        
        self.prev_test_loss = jnp.finfo(jnp.float32).max
        self.early_stopping_counter = 0
    
    def maybe_save_best_model(self, *args, **kwargs):
        raise NotImplementedError('Must be model-specific')
    
    def check_early_stop(self,
                         args,
                         epoch_loss):
        ### condition 1: if test loss stagnates or starts to go up, compared
        ###              to previous epoch's test loss
        cond1 = jnp.allclose( self.prev_test_loss, 
                              jnp.minimum (self.prev_test_loss, epoch_loss), 
                              atol=args.early_stop_cond1_atol,
                              rtol=0 )

        ### condition 2: if test loss is substatially worse than best test loss
        cond2 = (epoch_loss - self.best_test_loss) > args.early_stop_cond2_gap

        if cond1 or cond2:
            self.early_stopping_counter += 1
        else:
            self.early_stopping_counter = 0
        
        if self.early_stopping_counter == args.patience:
            return True
        
        else:
            return False

class timers:
    def __init__(self, 
                 num_epochs):
        self.num_epochs = num_epochs
        self.all_times = np.zeros( (self.num_epochs, 2) )
        self.cache = None
        
    def start_timer(self):
        real = wall_clock_time()
        cpu = process_time()
        self.cache = (real, cpu)
    
    def end_timer(self):
        real_start, cpu_start = self.cache
        real_end = wall_clock_time()
        cpu_end = process_time() 
        
        # get deltas
        real_delta = real_end - real_start
        cpu_delta = cpu_end - cpu_start
        
        # clear cache
        self.cache = None
        
        return (real_delta, cpu_delta)
        
    def end_timer_and_write_to_tboard(self, 
                                      epoch_idx,
                                      writer,
                                      tag ):
        real_delta, cpu_delta = end_timer()
        
        # record for later
        self.all_times[epoch_idx, 0] = real_delta
        self.all_times[epoch_idx, 1] = cpu_delta

        # write to tensorboard
        write_times(cpu_start = cpu_start, 
                    cpu_end = cpu_end, 
                    real_start = real_start, 
                    real_end = real_end, 
                    tag = tag, 
                    step = epoch_idx, 
                    writer_obj = writer)  
        
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

def pigz_compress_tensorboard_file( args ):
    print('\n\nDONE; compressing tboard folder')
    if platform.system() == 'Darwin':
        os.system(f"tar --use-compress-program=pigz -cf {args.training_wkdir}/tboard.tar.gz {args.training_wkdir}/tboard")
    
    elif platform.system() == 'Linux':
        os.system(f"tar --use-compress-program=pigz -cf {args.training_wkdir}/tboard.tar.gz --remove-files {args.training_wkdir}/tboard")
    
    
    