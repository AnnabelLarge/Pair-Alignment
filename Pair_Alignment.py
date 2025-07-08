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
import gc

from dloaders.init_dataloader import init_dataloader

# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_debug_infs", True)
jax.config.update("jax_enable_x64", True)


def main():
    if 'RESULTS' in os.listdir():
        shutil.rmtree('RESULTS')
    
    
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
                   'batched_train',
                   'continue_train',
                   'eval',
                   'batched_eval',
                   'prep_datasets',
                   'validate_neuraltkf_config',
                   'batched_validate_neuraltkf_config']
    
    # parser.add_argument('-task',
    #                     type=str,
    #                     required=True,
    #                     choices = valid_tasks,
    #                     help=f'What do you want to do? Pick from: {valid_tasks}')
    
    # parser.add_argument('-configs',
    #                     type = str,
    #                     required=True,
    #                     help='Load configs from file or folder of files, in json format.')
    
    # # optional: might have pre-processed some dataloaders
    # parser.add_argument('-load_dset_pkl',
    #                     type = str,
    #                     default=None,
    #                     help='name of the pre-computed pytorch dataset pickle object')
    
    # # only needed when continuing training
    # parser.add_argument('-new_training_wkdir',
    #                     type = str,
    #                     help='FOR CONTINUE_TRAIN OPTION; Name for a new training working dir')
    
    # parser.add_argument('-prev_model_ckpts_dir',
    #                     type = str,
    #                     help='FOR CONTINUE_TRAIN OPTION; Path to previous trainstate, argparse object')
    
    # parser.add_argument('-tstate_to_load',
    #                     type = str,
    #                     help='FOR CONTINUE_TRAIN OPTION; The name of the tstate object to load')
    
    # parse the arguments
    top_level_args = parser.parse_args()
    
    
    ## UNCOMMENT TO RUN IN SPYDER IDE
    top_level_args.task = 'train'
    top_level_args.configs = 'CONFIG.json'
    top_level_args.load_dset_pkl = None
    
    
    ### helper functions 
    # open a single config file and extract additional arguments
    def read_config_file(config_file):
        with open(config_file, 'r') as f:
            contents = json.load(f)
            t_args = argparse.Namespace()
            t_args.__dict__.update(contents)
            args = parser.parse_args(namespace=t_args)
        return args
    
    # load a pre-computed dataset; make a pytorch dataloader
    def load_dset_pkl_fn(file_to_load, 
                      args,
                      collate_fn):
        with open(file_to_load,'rb') as f:
            dset_dict = pickle.load(f)
        
        # add dataloader objects
        test_dl = init_dataloader(args = args, 
                                  shuffle = False,
                                  pytorch_custom_dset = dset_dict['test_dset'],
                                  collate_fn = collate_fn)
        dset_dict['test_dl'] = test_dl
        
        if 'training_dset' in dset_dict.keys():
            training_dl = init_dataloader(args = args, 
                                            shuffle = True,
                                            pytorch_custom_dset = dset_dict['training_dset'],
                                            collate_fn = collate_fn)
            dset_dict['training_dl'] = training_dl
        return dset_dict


    ###########################################################################
    ### TRAINING   ############################################################
    ###########################################################################
    if top_level_args.task == 'train':
        # read argparse
        assert top_level_args.configs.endswith('.json'), "input is one JSON file"
        print(f'TRAINING WITH: {top_level_args.configs}')
        args = read_config_file(top_level_args.configs)
        pred_model_type = args.pred_model_type
        
        # import correct wrappers, dataloader initializers
        if pred_model_type == 'pairhmm_indp_sites':
            from cli.train_pairhmm_indp_sites import train_pairhmm_indp_sites as train_fn
            from dloaders.init_counts_dset import init_counts_dset as init_datasets
            from dloaders.CountsDset import jax_collator as collate_fn
            
        elif pred_model_type in ['pairhmm_frag_and_site_classes',
                                      'neural_hmm',
                                      'feedforward']:
            from dloaders.init_full_len_dset import init_full_len_dset as init_datasets
            from dloaders.FullLenDset import jax_collator as collate_fn
            
            if pred_model_type == 'pairhmm_frag_and_site_classes':
                from cli.train_pairhmm_frag_and_site_classes import train_pairhmm_frag_and_site_classes as train_fn
                
            elif pred_model_type == 'neural_hmm':
                from cli.train_neural_hmm import train_neural_hmm as train_fn
    
            elif pred_model_type == 'feedforward':
                raise NotImplementedError('not ready')
                
        # make dataloder list
        if top_level_args.load_dset_pkl is None:
            dload_dict = init_datasets( args,
                                          'train',
                                          training_argparse = None,
                                          include_dataloader = True)
        else:
            dload_dict = load_dset_pkl_fn(args = args,
                                       file_to_load = top_level_args.load_dset_pkl,
                                       collate_fn = collate_fn)
            
        # train model
        train_fn( args, dload_dict )


    elif top_level_args.task == 'batched_train':
        # read argparse from first config file
        file_lst = [file for file in os.listdir(top_level_args.configs) if not file.startswith('.')
                    and file.endswith('.json')]
        assert len(file_lst) > 0, f'{top_level_args.configs} is empty!'
        
        # get dataloader and functions from first config file
        first_config_file = file_lst[0]
        print(f'DATALOADER CONSTRUCTED FROM: {top_level_args.configs}/{first_config_file}')
        print(f"WARNING: make sure you want this dataloader for ALL experiments in {top_level_args.configs}!!!")
        first_args = read_config_file(f'{top_level_args.configs}/{first_config_file}')
        pred_model_type = first_args.pred_model_type
        
        # import correct wrappers, dataloader initializers
        if pred_model_type == 'pairhmm_indp_sites':
            from cli.train_pairhmm_indp_sites import train_pairhmm_indp_sites as train_fn
            from dloaders.init_counts_dset import init_counts_dset as init_datasets
            from dloaders.CountsDset import jax_collator as collate_fn
            
        elif pred_model_type in ['pairhmm_frag_and_site_classes',
                                      'neural_hmm',
                                      'feedforward']:
            from dloaders.init_full_len_dset import init_full_len_dset as init_datasets
            from dloaders.FullLenDset import jax_collator as collate_fn
            
            if pred_model_type == 'pairhmm_frag_and_site_classes':
                from cli.train_pairhmm_frag_and_site_classes import train_pairhmm_frag_and_site_classes as train_fn
                
            elif pred_model_type == 'neural_hmm':
                from cli.train_neural_hmm import train_neural_hmm as train_fn
    
            elif pred_model_type == 'feedforward':
                raise NotImplementedError('not ready')

        # load data
        if top_level_args.load_dset_pkl is None:
            dload_dict = init_datasets( first_args,
                                          'train',
                                          training_argparse = None,
                                          include_dataloader = True)
        else:
            dload_dict = load_dset_pkl_fn(args = first_args,
                                       file_to_load = top_level_args.load_dset_pkl,
                                       collate_fn = collate_fn)
            
        # with this dload_dict, train using ALL config files
        for file in file_lst:
            # read argparse
            assert file.endswith('.json'), "input is one JSON file"
            this_run_args = read_config_file(f'{top_level_args.configs}/{file}')
            print(f'TRAINING WITH: {top_level_args.configs}/{file}')
            
            train_fn( this_run_args, 
                      dload_dict_for_all )
            
            del this_run_args
    
    
    elif top_level_args.task == 'continue_train':
        # read argparse
        assert top_level_args.configs.endswith('.json'), "input is one JSON file"
        print(f'CONTINUE TRAINING WITH: {top_level_args.configs}, IN NEW DIR {top_level_args.new_training_wkdir}')
        args_from_training_config = read_config_file(top_level_args.configs)
        pred_model_type = args_from_training_config.pred_model_type
        
        # import correct wrappers, dataloader initializers
        if pred_model_type == 'pairhmm_indp_sites':
            from cli.cont_training_pairhmm_indp_sites import cont_training_pairhmm_indp_sites as cont_train_fn
            from dloaders.init_counts_dset import init_counts_dset as init_datasets
            from dloaders.CountsDset import jax_collator as collate_fn
        
        else:
            raise NotImplementedError('Cannot continue training yet!')

        # make dataloader objects
        if top_level_args.load_dset_pkl is None:
            dload_dict = init_datasets( args_from_training_config,
                                          'train',
                                          training_argparse = None,
                                          include_dataloader = True )
        else:
            dload_dict = load_dset_pkl_fn(args = args_from_training_config,
                                       file_to_load = top_level_args.load_dset_pkl,
                                       collate_fn = collate_fn)
            
        # train model
        cont_train_fn( args=args_from_training_config, 
                        dataloader_dict=dload_dict,
                        new_training_wkdir=args.new_training_wkdir,
                        prev_model_ckpts_dir=args.prev_model_ckpts_dir,
                        tstate_to_load=args.tstate_to_load
                        )
    
    
    ###########################################################################
    ### EVAL   ################################################################
    ###########################################################################
    elif top_level_args.task == 'eval':
        # read argparse
        assert top_level_args.configs.endswith('.json'), "input is one JSON file"
        print(f'EVALUATING WITH: {top_level_args.configs}')
        args = read_config_file(top_level_args.configs)
        
        # find and read training argparse
        model_ckpts_dir = f'{os.getcwd()}/{args.training_wkdir}/model_ckpts'
        training_argparse_filename = model_ckpts_dir + '/' + 'TRAINING_ARGPARSE.pkl'
        
        with open(training_argparse_filename,'rb') as g:
            training_argparse = pickle.load(g)
        
        pred_model_type = training_argparse.pred_model_type
        
        # import correct wrappers, dataloader initializers
        if pred_model_type == 'pairhmm_indp_sites':
            from cli.eval_pairhmm_indp_sites import eval_pairhmm_indp_sites as eval_fn
            from dloaders.init_counts_dset import init_counts_dset as init_datasets
            from dloaders.CountsDset import jax_collator as collate_fn

        elif pred_model_type in ['pairhmm_frag_and_site_classes',
                                 'neural_hmm',
                                 'feedforward']:
            from dloaders.init_full_len_dset import init_full_len_dset as init_datasets
            from dloaders.FullLenDset import jax_collator as collate_fn

            if pred_model_type == 'pairhmm_frag_and_site_classes':
                from cli.eval_pairhmm_frag_and_site_classes import eval_pairhmm_frag_and_site_classes as eval_fn

            elif pred_model_type == 'neural_hmm':
                from cli.eval_neural_hmm import eval_neural_hmm as eval_fn
            
            elif pred_model_type == 'feedforward':
                raise NotImplementedError('not ready')

        # load data; saved under trianing_wkdir name
        if top_level_args.load_dset_pkl is None:
            dload_dict = init_datasets( args,
                                        'eval',
                                        training_argparse,
                                        include_dataloader = True )
        else:
            dload_dict = load_dset_pkl_fn(args = args,
                                       file_to_load = top_level_args.load_dset_pkl,
                                       collate_fn = collate_fn)
            
        # evaluate model
        eval_fn( args, 
                  dload_dict, 
                  training_argparse )
    
    
    elif top_level_args.task == 'batched_eval':
        # read argparse from first config file
        file_lst = [file for file in os.listdir(top_level_args.configs) if not file.startswith('.')
                    and file.endswith('.json')]
        assert len(file_lst) > 0, f'{top_level_args.configs} is empty!'
        
        # get dataloader and functions from first config file
        first_config_file = file_lst[0]
        print(f'DATALOADER CONSTRUCTED FROM: {top_level_args.configs}/{first_config_file}')
        print(f"WARNING: make sure you want this dataloader for ALL experiments in {top_level_args.configs}!!!")
        first_args = read_config_file(f'{top_level_args.configs}/{first_config_file}')
        
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
            from dloaders.init_counts_dset import init_counts_dset as init_datasets
            from dloaders.CountsDset import jax_collator as collate_fn

        elif pred_model_type in ['pairhmm_frag_and_site_classes',
                                 'neural_hmm',
                                 'feedforward']:
            from dloaders.init_full_len_dset import init_full_len_dset as init_datasets
            from dloaders.FullLenDset import jax_collator as collate_fn

            if pred_model_type == 'pairhmm_frag_and_site_classes':
                from cli.eval_pairhmm_frag_and_site_classes import eval_pairhmm_frag_and_site_classes as eval_fn

            elif pred_model_type == 'neural_hmm':
                from cli.eval_neural_hmm import eval_neural_hmm as eval_fn
            
            elif pred_model_type == 'feedforward':
                raise NotImplementedError('not ready')
            
        # load data
        if top_level_args.load_dset_pkl is None:
            dload_dict_for_all = init_datasets( first_args,
                                          'eval',
                                          first_training_argparse,
                                          include_dataloader = True)
        else:
            dload_dict_for_all = load_dset_pkl_fn(args = first_args,
                                       file_to_load = top_level_args.load_dset_pkl,
                                       collate_fn = collate_fn)
            
        del first_training_argparse, first_args
        
        # with this dload_dict, train using ALL config files
        for file in file_lst:
            # read argparse
            assert file.endswith('.json'), "input is one JSON file"
            this_run_args = read_config_file(f'{top_level_args.configs}/{file}')
            print(f'EVALUATING WITH: {top_level_args.configs}/{file}')
            
            # find and read training argparse
            training_argparse_filename = (f'{os.getcwd()}/'+
                                          f'{this_run_args.training_wkdir}/'+
                                          f'model_ckpts/'+
                                          f'TRAINING_ARGPARSE.pkl')
            
            with open(training_argparse_filename,'rb') as g:
                training_argparse = pickle.load(g)
            
            eval_fn( this_run_args, 
                     dload_dict_for_all, 
                     training_argparse )
            
            del this_run_args, training_argparse
    
    
    # ###########################################################################
    # ### EVAL: label class posterior marginals for markovian class sites   #####
    # ###########################################################################
    # elif top_level_args.task == 'label_class_post':
    #     raise NotImplementedError('not ready yet')
    #     ### read argparse
    #     assert top_level_args.configs.endswith('.json'), "input is one JSON file"
    #     print(f'EVALUATING WITH: {top_level_args.configs}')
    #     args = read_config_file(top_level_args.configs)
        
        
    #     ### find and read training argparse
    #     model_ckpts_dir = f'{os.getcwd()}/{args.training_wkdir}/model_ckpts'
    #     training_argparse_filename = model_ckpts_dir + '/' + 'TRAINING_ARGPARSE.pkl'
        
    #     with open(training_argparse_filename,'rb') as g:
    #         training_argparse = pickle.load(g)
        
    #     pred_model_type = training_argparse.pred_model_type
        
        
    #     ### import correct wrappers, dataloader initializers
    #     from cli.class_posteriors_pairhmm_frag_and_site_classes import class_posteriors_pairhmm_frag_and_site_classes as labeling_fn
    #     from dloaders.init_full_len_dset import init_full_len_dset as init_datasets
    #     from dloaders.FullLenDset import jax_collator as collate_fn

    #     # load data (saved under training_wkdir name)
    #     if top_level_args.load_dset_pkl is None:
    #         dload_dict = init_datasets( args,
    #                                       'eval',
    #                                       training_argparse,
    #                                       include_dataloader = True )
    #     else:
    #         dload_dict = load_dset_pkl_fn(args = args,
    #                                    file_to_load = top_level_args.load_dset_pkl,
    #                                    collate_fn = collate_fn)
        
    #     # evaluate
    #     labeling_fn( args, 
    #                  dload_dict, 
    #                  training_argparse )
    
    
    ###########################################################################
    ### MISC: prepare dataloaders for downstream use   ########################
    ###########################################################################
    elif top_level_args.task == 'prep_datasets':
        file_lst = [file for file in os.listdir(top_level_args.configs) if not file.startswith('.')
                    and file.endswith('.json')]
        assert len(file_lst) > 0, f'{top_level_args.configs} is empty!'
        
        
        ### get modules from first config file
        first_config_file = file_lst[0]
        first_args = read_config_file(f'{top_level_args.configs}/{first_config_file}')
        
        # import correct wrappers, dataloader initializers
        if first_args.pred_model_type == 'pairhmm_indp_sites':
            from cli.train_pairhmm_indp_sites import train_pairhmm_indp_sites as train_fn
            from dloaders.init_counts_dset import init_counts_dset as init_datasets

        elif first_args.pred_model_type in ['pairhmm_frag_and_site_classes',
                                            'neural_hmm',
                                            'feedforward']:
            from cli.train_pairhmm_frag_and_site_classes import train_pairhmm_frag_and_site_classes as train_fn
            from dloaders.init_full_len_dset import init_full_len_dset as init_datasets
            
        del first_config_file, first_args
        
        
        ### read every file, then export a raw pickle 
        for file in file_lst:
            # read argparse
            assert file.endswith('.json'), "input is one JSON file"
            this_run_args = read_config_file(f'{top_level_args.configs}/{file}')
            print(f'LOADING FROM: {top_level_args.configs}/{file}')
            
            # determine if four or two dataloaders should be returned
            checklist = ['train_dset_splits', 'test_dset_splits', 'optimizer_config']
            train_flag = all(arg in dir(this_run_args) for arg in checklist)
            
            # load
            dload_dict = init_datasets( this_run_args, 
                                          'train' if train_flag else 'eval',
                                          training_argparse = None,
                                          include_dataloader = False)
            
            # dump; use the config name by default
            new_file_name = f'TMP-dload-lst_' + file.replace('.json','.pkl')
            print(f'SAVING TO: {new_file_name}')
            with open(new_file_name, 'wb') as g:
                pickle.dump(dload_dict, g)

            del this_run_args, checklist, train_flag, dload_dict, new_file_name
        
        print('done')
    
    
    ###########################################################################
    ### CONFIG CHECK: Validate Neural TKF config file   #######################
    ###########################################################################
    # 'validate_neuraltkf_config',
    # 'batched_validate_neuraltkf_config'
    elif top_level_args.task == 'validate_neuraltkf_config':
        from cli.test_neural_tkf_model_is_causal import test_neural_tkf_model_is_causal 
        assert top_level_args.configs.endswith('.json'), "input is one JSON file"
        
        test_neural_tkf_model_is_causal(top_level_args.configs)
        
    
    elif top_level_args.task == 'batched_validate_neuraltkf_config':
        from cli.test_neural_tkf_model_is_causal import test_neural_tkf_model_is_causal 
        
        file_lst = [file for file in os.listdir(top_level_args.configs) if not file.startswith('.')
                    and file.endswith('.json')]
        assert len(file_lst) > 0, f'{top_level_args.configs} is empty!'
        
        for file in file_lst:
            path = f'{top_level_args.configs}/{file}'
            test_neural_tkf_model_is_causal(path)
        
if __name__ == '__main__':
    main()