#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 16:13:55 2023

@author: annabel_large

"""
import json
import os
import argparse
import jax


# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_debug_infs", True)
# jax.config.update("jax_enable_x64", True)


def main():
    ### for now, running models on single GPU
    err_ms = 'SELECT GPU TO RUN THIS COMPUTATION ON with CUDA_VISIBLE_DEVICES=DEVICE_NUM'
    assert len(jax.devices()) == 1, err_ms
    del err_ms
    
    ###########################################################################
    ### INITIALIZE PARSER   ###################################################
    ###########################################################################
    parser = argparse.ArgumentParser(prog='Pair_Alignment')
    
    ### which program do you want to run?
    valid_tasks = ['train',
                   'DEBUG_markovian_pairhmm',
                   'get_pairhmm_controls',
                   'get_feedforward_controls']
    
    # parser.add_argument('-task',
    #                     type=str,
    #                     required=True,
    #                     choices = valid_tasks,
    #                     help='What do you want to do? Pick from: {valid_tasks}')
    
    # # needed for most options
    # parser.add_argument('-configs',
    #                     type = str,
    #                     help='Load configs from file or folder of files, in json format.')
    
    # # only when resuming training
    # parser.add_argument(f'-training_wkdir',
    #                     type = str,
    #                     help = 'training working directory to resume from')
    
    # parse the arguments
    args = parser.parse_args()
    
    
    ### UNCOMMENT TO RUN IN SPYDER IDE
    args.task = 'DEBUG_markovian_pairhmm'
    args.configs = 'markovian_one_class.json'
    
    
    ### helper function to open a single config file and extract additional arguments
    def read_config_file(config_file):
        with open(config_file, 'r') as f:
            contents = json.load(f)
            t_args = argparse.Namespace()
            t_args.__dict__.update(contents)
            args = parser.parse_args(namespace=t_args)
        return args


    ###########################################################################
    ### TRAINING   ############################################################
    ###########################################################################
    if args.task == 'train':
        # read argparse
        assert args.configs.endswith('.json'), print("input is one JSON file")
        print(f'TRAINING WITH: {args.configs}')
        args = read_config_file(args.configs)
        
        # import correct wrappers, dataloader initializers
        if args.pred_model_type == 'pairhmm_indp_sites':
            from cli.train_pairhmm_indp_sites import train_pairhmm_indp_sites as train
            from dloaders.init_counts_dset import init_counts_dset as init_dataloaders

        elif args.pred_model_type == 'pairhmm_markovian_sites':
            from cli.train_pairhmm_markovian_sites import train_pairhmm_markovian_sites as train
            from dloaders.init_full_len_dset import init_full_len_dset as init_dataloaders

        elif args.pred_model_type == 'neural_hmm':
            from cli.train_neural_hmm import train_neural_hmm as train
            from dloaders.init_full_len_dset import init_full_len_dset as init_dataloaders

        elif args.pred_model_type == 'feedforward':
            from cli.train_feedforward import train_feedforward as train
            from dloaders.init_full_len_dset import init_full_len_dset as init_dataloaders

        # train model
        dload_lst = init_dataloaders(args, 'train')
        train(args, dload_lst)
    
    
    
    elif args.task == 'DEBUG_markovian_pairhmm':
        assert args.configs.endswith('.json'), print("input is one JSON file")
        print(f'TRAINING WITH: {args.configs}')
        args = read_config_file(args.configs)
        
        from dloaders.init_full_len_dset import init_full_len_dset as init_dataloaders
        from cli.DEBUG_markovian_pairhmm import train_pairhmm_markovian_sites_pure_jax as debug_train
        
        dload_lst = init_dataloaders(args, 'train')
        debug_train(args, dload_lst)
        
        
    
    ###########################################################################
    ### GET CONTROLS   ########################################################
    ###########################################################################
    # for pairhmm and neural_hmm: compare score to pairhmm where
    #   transition and emission matrices are derived from counts directly
    elif args.task == 'get_pairhmm_controls':
        from cli.get_pairhmm_controls import get_pairhmm_controls
        get_pairhmm_controls(args)
    
    # for feedforward seq2seq (conditional loss): compare to naive 
    #   frequency-based counts
    elif args.task == 'get_feedforward_controls':
        from cli.get_feedforward_controls import get_feedforward_controls
        get_feedforward_controls(args)
    
    
    
if __name__ == '__main__':
    main()