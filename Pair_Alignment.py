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

# functions to run
from cli.train import train
from dloaders.init_dataloaders import init_dataloaders


# jax.config.update("jax_debug_nans", True)
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
    valid_tasks = ['train']
    
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
    
    
    # ### UNCOMMENT TO RUN IN SPYDER IDE
    args.task = 'train'
    args.configs = 'tkf91_load_params.json'
    
    
    ### helper function to open a single config file and extract additional arguments
    def read_config_file(config_file):
        with open(config_file, 'r') as f:
            contents = json.load(f)
            
            t_args = argparse.Namespace()
            t_args.__dict__.update(contents)
            args = parser.parse_args(namespace=t_args)
        return args


    ###########################################################################
    ### TRAINING OPTIONS   ####################################################
    ###########################################################################

    ############
    ### TRAIN  #
    ############
    if args.task == 'train':
        assert args.configs.endswith('.json'), print("input is one JSON file")
        print(f'TRAINING WITH: {args.configs}')
        args = read_config_file(args.configs)
        
        dload_lst = init_dataloaders(args, 'train')
        train(args, dload_lst)
    
    

if __name__ == '__main__':
    import shutil
    import os
    
    folder = 'RESULTS_tkf91_load_params'
    print(f'REMOVING PREVIOUS RUN: {folder}')
    
    if folder in os.listdir():
        shutil.rmtree(folder)


    main()
