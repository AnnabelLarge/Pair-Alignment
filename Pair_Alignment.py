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
import pickle
import shutil

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
                    'eval',
                    'batched_train',
                    'batched_eval',
                    'label_class_post',
                    'continue_train']
    
    parser.add_argument('-task',
                        type=str,
                        required=True,
                        choices = valid_tasks,
                        help='What do you want to do? Pick from: {valid_tasks}')
    
    parser.add_argument('-configs',
                        type = str,
                        required=True,
                        help='Load configs from file or folder of files, in json format.')
    
    # only needed when continuing training
    parser.add_argument('-new_training_wkdir',
                        type = str,
                        help='ONLY FOR CONTINUE_TRAIN OPTION; Name for a new training working dir')
    
    parser.add_argument('-prev_model_ckpts_dir',
                        type = str,
                        help='ONLY FOR CONTINUE_TRAIN OPTION; Path to previous trainstate, argparse object')
    
    parser.add_argument('-tstate_to_load',
                        type = str,
                        help='ONLY FOR CONTINUE_TRAIN OPTION; The name of the tstate object to load')
    
    # parse the arguments
    args = parser.parse_args()
    
    
    # ## UNCOMMENT TO RUN IN SPYDER IDE
    # args.task = 'train'
    # args.configs = 'CONFIG_DRY-RUN_gtr_60-indp-site.json'
    
    
    
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
            from cli.train_pairhmm_indp_sites import train_pairhmm_indp_sites as train_fn
            from dloaders.init_counts_dset import init_counts_dset as init_dataloaders

        elif args.pred_model_type == 'pairhmm_markovian_sites':
            from cli.train_pairhmm_markovian_sites import train_pairhmm_markovian_sites as train_fn
            from dloaders.init_full_len_dset import init_full_len_dset as init_dataloaders

        elif args.pred_model_type == 'neural_hmm':
            from cli.train_neural_hmm import train_neural_hmm as train_fn
            from dloaders.init_full_len_dset import init_full_len_dset as init_dataloaders

        elif args.pred_model_type == 'feedforward':
            from cli.train_feedforward import train_feedforward as train_fn
            from dloaders.init_full_len_dset import init_full_len_dset as init_dataloaders

        # train model
        dload_lst = init_dataloaders( args,
                                      'train',
                                      training_argparse = None )
        
        train_fn( args, 
                  dload_lst )

    elif args.task == 'batched_train':
        ### read argparse from first config file
        file_lst = [file for file in os.listdir(args.configs) if not file.startswith('.')
                    and file.endswith('.json')]
        assert len(file_lst) > 0, f'{args.configs} is empty!'
        
        
        ### get dataloader and functions from first config file
        first_config_file = file_lst[0]
        print(f'DATALOADER CONSTRUCTED FROM: {args.configs}/{first_config_file}')
        print(f"WARNING: make sure you want this dataloader for ALL experiments in {args.configs}!!!")
        print(f"WARNING: make sure you want this dataloader for ALL experiments in {args.configs}!!!")
        print(f"WARNING: make sure you want this dataloader for ALL experiments in {args.configs}!!!")
        first_args = read_config_file(f'{args.configs}/{first_config_file}')
        
        # import correct wrappers, dataloader initializers
        if first_args.pred_model_type == 'pairhmm_indp_sites':
            from cli.train_pairhmm_indp_sites import train_pairhmm_indp_sites as train_fn
            from dloaders.init_counts_dset import init_counts_dset as init_dataloaders

        elif first_args.pred_model_type == 'pairhmm_markovian_sites':
            from cli.train_pairhmm_markovian_sites import train_pairhmm_markovian_sites as train_fn
            from dloaders.init_full_len_dset import init_full_len_dset as init_dataloaders

        elif first_args.pred_model_type == 'neural_hmm':
            from cli.train_neural_hmm import train_neural_hmm as train_fn
            from dloaders.init_full_len_dset import init_full_len_dset as init_dataloaders

        elif first_args.pred_model_type == 'feedforward':
            from cli.train_feedforward import train_feedforward as train_fn
            from dloaders.init_full_len_dset import init_full_len_dset as init_dataloaders

        # load data
        dload_lst_for_all = init_dataloaders( first_args, 
                                             'train',
                                             training_argparse = None  )
        
        del first_args
        
        
        ### with this dload_lst, train using ALL config files
        for file in file_lst:
            # read argparse
            assert file.endswith('.json'), print("input is one JSON file")
            this_run_args = read_config_file(f'{args.configs}/{file}')
            print(f'TRAINING WITH: {args.configs}/{file}')
            
            train_fn( this_run_args, 
                      dload_lst_for_all )
            
            del this_run_args
    
    elif args.task == 'continue_train':
        # read argparse
        assert args.configs.endswith('.json'), print("input is one JSON file")
        print(f'CONTINUE TRAINING WITH: {args.configs}, IN NEW DIR {args.new_training_wkdir}')
        args_from_training_config = read_config_file(args.configs)
        
        # import correct wrappers, dataloader initializers
        if args_from_training_config.pred_model_type == 'pairhmm_indp_sites':
            from cli.cont_training_pairhmm_indp_sites import cont_training_pairhmm_indp_sites as cont_train_fn
            from dloaders.init_counts_dset import init_counts_dset as init_dataloaders
        
        else:
            raise NotImplementedError('Cannot continue training yet!')
        

        # train model
        dload_lst = init_dataloaders( args_from_training_config,
                                      'train',
                                      training_argparse = None )
        
        cont_train_fn( args=args_from_training_config, 
                       dataloader_dict=dload_lst,
                       new_training_wkdir=args.new_training_wkdir,
                       prev_model_ckpts_dir=args.prev_model_ckpts_dir,
                       tstate_to_load=args.tstate_to_load
                       )
      
    
    ###########################################################################
    ### EVAL   ################################################################
    ###########################################################################
    elif args.task == 'eval':
        ### read argparse
        assert args.configs.endswith('.json'), print("input is one JSON file")
        print(f'EVALUATING WITH: {args.configs}')
        args = read_config_file(args.configs)
        
        
        ### find and read training argparse
        model_ckpts_dir = f'{os.getcwd()}/{args.training_wkdir}/model_ckpts'
        training_argparse_filename = model_ckpts_dir + '/' + 'TRAINING_ARGPARSE.pkl'
        
        with open(training_argparse_filename,'rb') as g:
            training_argparse = pickle.load(g)
        
        pred_model_type = training_argparse.pred_model_type
        
        
        ### import correct wrappers, dataloader initializers
        if pred_model_type == 'pairhmm_indp_sites':
            from cli.eval_pairhmm_indp_sites import eval_pairhmm_indp_sites as eval_fn
            from dloaders.init_counts_dset import init_counts_dset as init_dataloaders

        elif pred_model_type == 'pairhmm_markovian_sites':
            from cli.eval_pairhmm_markovian_sites import eval_pairhmm_markovian_sites as eval_fn
            from dloaders.init_full_len_dset import init_full_len_dset as init_dataloaders

        # evaluate model
        dload_lst = init_dataloaders( args, 
                                      'eval',
                                      training_argparse )
        
        eval_fn( args, 
                 dload_lst, 
                 training_argparse )
    
    
    elif args.task == 'batched_eval':
        ### read argparse from first config file
        file_lst = [file for file in os.listdir(args.configs) if not file.startswith('.')
                    and file.endswith('.json')]
        assert len(file_lst) > 0, f'{args.configs} is empty!'
        
        
        ### get dataloader and functions from first config file
        first_config_file = file_lst[0]
        print(f'DATALOADER CONSTRUCTED FROM: {args.configs}/{first_config_file}')
        print(f"WARNING: make sure you want this dataloader for ALL experiments in {args.configs}!!!")
        print(f"WARNING: make sure you want this dataloader for ALL experiments in {args.configs}!!!")
        print(f"WARNING: make sure you want this dataloader for ALL experiments in {args.configs}!!!")
        first_args = read_config_file(f'{args.configs}/{first_config_file}')
        
        # find and read training argparse
        first_training_argparse_filename = (f'{os.getcwd()}/'+
                                            f'{first_args.training_wkdir}/'+
                                            f'model_ckpts/'+
                                            f'TRAINING_ARGPARSE.pkl')
        
        with open(first_training_argparse_filename,'rb') as g:
            first_training_argparse = pickle.load(g)
        
        pred_model_type = first_training_argparse.pred_model_type
        
        # import correct wrappers, dataloader initializers
        if pred_model_type == 'pairhmm_indp_sites':
            from cli.eval_pairhmm_indp_sites import eval_pairhmm_indp_sites as eval_fn
            from dloaders.init_counts_dset import init_counts_dset as init_dataloaders

        elif pred_model_type == 'pairhmm_markovian_sites':
            from cli.eval_pairhmm_markovian_sites import eval_pairhmm_markovian_sites as eval_fn
            from dloaders.init_full_len_dset import init_full_len_dset as init_dataloaders

        # load data
        dload_lst_for_all = init_dataloaders( first_args, 
                                             'eval',
                                             first_training_argparse )
        
        del first_training_argparse, first_args
        
        
        ### with this dload_lst, train using ALL config files
        for file in file_lst:
            # read argparse
            assert file.endswith('.json'), print("input is one JSON file")
            this_run_args = read_config_file(f'{args.configs}/{file}')
            print(f'EVALUATING WITH: {args.configs}/{file}')
            
            # find and read training argparse
            training_argparse_filename = (f'{os.getcwd()}/'+
                                          f'{this_run_args.training_wkdir}/'+
                                          f'model_ckpts/'+
                                          f'TRAINING_ARGPARSE.pkl')
            
            with open(training_argparse_filename,'rb') as g:
                training_argparse = pickle.load(g)
            
            eval_fn( this_run_args, 
                     dload_lst_for_all, 
                     training_argparse )
            
            del this_run_args, training_argparse
    
    
    ###########################################################################
    ### EVAL: label class posterior marginals for markovian class sites   #####
    ###########################################################################
    elif args.task == 'label_class_post':
        ### read argparse
        assert args.configs.endswith('.json'), print("input is one JSON file")
        print(f'EVALUATING WITH: {args.configs}')
        args = read_config_file(args.configs)
        
        
        ### find and read training argparse
        model_ckpts_dir = f'{os.getcwd()}/{args.training_wkdir}/model_ckpts'
        training_argparse_filename = model_ckpts_dir + '/' + 'TRAINING_ARGPARSE.pkl'
        
        with open(training_argparse_filename,'rb') as g:
            training_argparse = pickle.load(g)
        
        pred_model_type = training_argparse.pred_model_type
        
        
        ### import correct wrappers, dataloader initializers
        from cli.class_posteriors_pairhmm_markovian_sites import class_posteriors_pairhmm_markovian_sites as labeling_fn
        from dloaders.init_full_len_dset import init_full_len_dset as init_dataloaders

        # evaluate
        dload_lst = init_dataloaders( args, 
                                      'eval',
                                      training_argparse )
        
        labeling_fn( args, 
                     dload_lst, 
                     training_argparse )
        
    
if __name__ == '__main__':
    main()