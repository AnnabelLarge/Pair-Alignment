#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 17:44:15 2025

@author: annabel
"""

### might be able to use this across ALL models
def write_final_eval_results(args,
                             summary_stats: dict,
                             filename: str):
    to_write_prefix = {'RUN': args.training_wkdir}
    to_write = {**to_write_prefix, **summary_stats}
    
    with open(f'{args.logfile_dir}/{filename}','w') as g:
        for k, v in to_write.items():
            g.write(f'{k}\t{v}\n')
        


### handle time
# write final timing
write_timing_file( outdir = args.logfile_dir,
                   train_times = all_train_set_times,
                   eval_times = all_eval_set_times,
                   total_times = all_epoch_times )

del all_train_set_times, all_eval_set_times, all_epoch_times

# new timer
postproc_timer_class = timers( num_epochs = args.num_epochs )


### write to output logfile
with open(args.logfile_name,'a') as g:
    # if early stopping was never triggered, record results at last epoch
    if not early_stop:
        g.write(f'Regular stopping after {epoch_idx} full epochs:\n\n')
    
    # finish up logfile, regardless of early stopping or not
    g.write(f'Epoch with lowest average test loss ("best epoch"): {best_epoch}\n')
    g.write(f'RE-EVALUATING ALL DATA WITH BEST PARAMS\n\n')

del epoch_idx


### save the argparse object by itself
args.epoch_idx = best_epoch
with open(f'{args.model_ckpts_dir}/TRAINING_ARGPARSE.pkl', 'wb') as g:
    pickle.dump(args, g)


### jit compile new eval function
# if this is a transformer model, will have extra arguments for eval funciton
extra_args_for_eval = dict()
if (args.anc_model_type == 'transformer' and args.desc_model_type == 'transformer'):
    flag = (args.anc_enc_config.get('output_attn_weights',False) or 
            args.desc_dec_config.get('output_attn_weights',False))
    extra_args_for_eval['output_attn_weights'] = flag

parted_eval_fn = partial( eval_one_batch,
                          all_model_instances = all_model_instances,
                          interms_for_tboard = args.interms_for_tboard,
                          t_array_for_all_samples = t_array_for_all_samples,  
                          concat_fn = concat_fn,
                          norm_loss_by_for_reporting = args.norm_reported_loss_by,  
                          extra_args_for_eval = extra_args_for_eval )
del extra_args_for_eval

# jit compile this eval function
eval_fn_jitted = jax.jit( parted_eval_fn, 
                          static_argnames = ['max_seq_len', 'max_align_len'])
del parted_eval_fn

###########################################
### loop through training dataloader and  #
### score with best params                #
###########################################
with open(args.logfile_name,'a') as g:
    g.write(f'SCORING ALL TRAIN SEQS\n')
    
train_summary_stats = final_eval_wrapper(dataloader = training_dl, 
                                         dataset = training_dset, 
                                         best_trainstates = best_trainstates, 
                                         jitted_determine_seqlen_bin = jitted_determine_seqlen_bin,
                                         jitted_determine_alignlen_bin = jitted_determine_alignlen_bin,
                                         eval_fn_jitted = eval_fn_jitted,
                                         out_alph_size = args.full_alphabet_size,
                                         save_arrs = args.save_arrs,
                                         save_per_sample_losses = args.save_per_sample_losses,
                                         interms_for_tboard = args.interms_for_tboard, 
                                         logfile_dir = args.logfile_dir,
                                         out_arrs_dir = args.out_arrs_dir,
                                         outfile_prefix = f'train-set',
                                         tboard_writer = writer)


###########################################
### loop through test dataloader and      #
### score with best params                #
###########################################
with open(args.logfile_name,'a') as g:
    g.write(f'SCORING ALL TEST SEQS\n')
    
# output_attn_weights also controlled by cond1 and cond2
test_summary_stats = final_eval_wrapper(dataloader = test_dl, 
                                         dataset = test_dset, 
                                         best_trainstates = best_trainstates, 
                                         jitted_determine_seqlen_bin = jitted_determine_seqlen_bin,
                                         jitted_determine_alignlen_bin = jitted_determine_alignlen_bin,
                                         eval_fn_jitted = eval_fn_jitted,
                                         out_alph_size = args.full_alphabet_size, 
                                         save_arrs = args.save_arrs,
                                         save_per_sample_losses = args.save_per_sample_losses,
                                         interms_for_tboard = args.interms_for_tboard, 
                                         logfile_dir = args.logfile_dir,
                                         out_arrs_dir = args.out_arrs_dir,
                                         outfile_prefix = f'test-set',
                                         tboard_writer = writer)


###########################################
### update the logfile with final losses  #
###########################################
write_final_eval_results(args = args, 
                         summary_stats = train_summary_stats,
                         filename = 'TRAIN-AVE-LOSSES.tsv')

write_final_eval_results(args = args, 
                         summary_stats = test_summary_stats,
                         filename = 'TEST-AVE-LOSSES.tsv')

# record total time spent on post-training actions; write this to a table
#   instead of a scalar collection
real_time, cpu_sys_time = postproc_timer_class.end_timer()
df = pd.DataFrame({'label': ['Real time', 'CPU+sys time'],
                   'value': [real_time, cpu_sys_time]})
markdown_table = df.to_markdown()
writer.add_text(tag = 'Code Timing | Post-training actions',
                text_string = markdown_table,
                global_step = 0)

# when you're done with the function, close the tensorboard writer and
#   compress the output file
writer.close()

# don't remove source on macOS (when I'm doing CPU testing)
print('\n\nDONE; compressing tboard folder')
if platform.system() == 'Darwin':
    os.system(f"tar -czvf {args.training_wkdir}/tboard.tar.gz {args.training_wkdir}/tboard")

# DO remove source on linux (when I'm doing real experiments)
elif platform.system() == 'Linux':
    os.system(f"tar -czvf {args.training_wkdir}/tboard.tar.gz  --remove-files {args.training_wkdir}/tboard")    
