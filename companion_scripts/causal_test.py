#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 20:02:45 2025

@author: annabel

Test all module parts for causality
"""
import jax
from jax import numpy as jnp



###############################################################################
### SEQUENCE EMBEDDERS   ######################################################
###############################################################################
# B = 1, L = 7
key = jax.random.key(0)
unaligned_seq = jnp.array( [[1,3,4,6,2,0]] )

###################################
### initial embedding functions   #
###################################
from models.sequence_embedders.initial_embedding_blocks import ( EmbeddingWithPadding,
                                                                   PlaceholderEmbedding ,
                                                                   TAPEEmbedding,
                                                                   ConvEmbedding )
def test_initial_embedding_blocks(model, variable_dict):
    whole_seq_result,_ = model.apply( datamat=unaligned_seq,
                                      variables=variable_dict,
                                      training=False)
    per_site_result = []
    for l in range(1,unaligned_seq.shape[1]+1):
        clipped_input = unaligned_seq[:,:l]
        last_site_result,_ = model.apply( datamat=clipped_input,
                                          variables=variable_dict,
                                          training=False )
        last_site_result = last_site_result[:,-1,:][:,None,:]
        per_site_result.append(last_site_result)
    per_site_result = jnp.concatenate( per_site_result, axis=1 )
    
    assert jnp.allclose(per_site_result, whole_seq_result)

    
model = PlaceholderEmbedding( config = {'hidden_dim': 10},
                              name = 'mod',
                              causal = True )
test_initial_embedding_blocks(model, variable_dict = {})
del model


model = EmbeddingWithPadding( config = {'hidden_dim': 10},
                              name = 'mod',
                              causal = True )
init_params = model.init( rngs=key,
                          datamat=jnp.zeros( unaligned_seq.shape,dtype=int ) )
test_initial_embedding_blocks(model, variable_dict = init_params)
del model, init_params


model = TAPEEmbedding( config = {'hidden_dim': 10},
                        name = 'mod',
                        causal = True )
init_params = model.init( rngs=key,
                          datamat=jnp.zeros( unaligned_seq.shape,dtype=int ),
                          training=False )
test_initial_embedding_blocks(model, variable_dict = init_params)
del model, init_params


for conv_kern_size in [1, 3, 20]:
    model = ConvEmbedding( config = {'base_alphabet_size': 23,
                                      'conv_emb_kernel_size': conv_kern_size,
                                      'hidden_dim': 10},
                            name = 'mod',
                            causal = True )
    init_params = model.init( rngs=key,
                              datamat=jnp.zeros( unaligned_seq.shape,dtype=int ),
                              training=False )
    test_initial_embedding_blocks(model, variable_dict = init_params)
    del model, init_params, conv_kern_size

    
############
### CNNs   #
############
def test_neural_embedding_blocks(model, variable_dict):
    whole_seq_result = model.apply( datamat=unaligned_seq,
                                      variables=variable_dict,
                                      training=False,
                                      sow_intermediates=False)
    if type(whole_seq_result) == tuple:
        whole_seq_result = whole_seq_result[1]
        
    per_site_result = []
    for l in range(1,unaligned_seq.shape[1]+1):
        clipped_input = unaligned_seq[:,:l]
        last_site_result = model.apply( datamat=clipped_input,
                                          variables=variable_dict,
                                          training=False,
                                          sow_intermediates=False )
        if type(last_site_result) == tuple:
            last_site_result = last_site_result[1]
        
        last_site_result = last_site_result[:,-1,:][:,None,:]
        per_site_result.append(last_site_result)
    per_site_result = jnp.concatenate( per_site_result, axis=1 )
    
    assert jnp.allclose(per_site_result, whole_seq_result)
    
    
from models.sequence_embedders.cnn.embedders import CNNSeqEmb
from models.sequence_embedders.cnn.blocks_fns import ConvnetBlock

config = {'kern_size_lst': [1,3,5],
          'hidden_dim': 10}
model = CNNSeqEmb( initial_embed_module = EmbeddingWithPadding,
                    first_block_module = ConvnetBlock,
                    subsequent_block_module = ConvnetBlock,
                    causal = True,
                    config = config,
                    name = 'mod' )
init_params = model.init( rngs=key,
                          datamat=jnp.zeros(unaligned_seq.shape,dtype=int),
                          training=False,
                          sow_intermediates=False)
test_neural_embedding_blocks(model, variable_dict=init_params)
del config, model, init_params


#############
### LSTMs   #
#############
from models.sequence_embedders.lstm.embedders import LSTMSeqEmb
from models.sequence_embedders.lstm.blocks_fns import UnidirecLSTMLayer

for n in [1, 3]:
    config = {'n_layers': n,
              'hidden_dim': 10}
    model = LSTMSeqEmb( initial_embed_module = EmbeddingWithPadding,
                        first_block_module = UnidirecLSTMLayer,
                        subsequent_block_module = UnidirecLSTMLayer,
                        causal = True,
                        config = config,
                        name = 'mod' )
    init_params = model.init( rngs=key,
                              datamat=jnp.zeros(unaligned_seq.shape,dtype=int),
                              training=False,
                              sow_intermediates=False)
    test_neural_embedding_blocks(model, variable_dict=init_params)
    del config, model, init_params, n


##################
## Transformers  #
##################
from models.sequence_embedders.transformer.embedders import TransfSeqEmb
from models.sequence_embedders.transformer.blocks_fns import (TransfBaseBlock,
                                                              RoPETransfBlock,
                                                              TransfBaseBlockWithAbsPosEmbedding,
                                                              TapeTransfBlock)

for n in [1, 3]:
    config = {'num_heads': 2,
              'num_blocks': n,
              'hidden_dim': 10}
    model = TransfSeqEmb( initial_embed_module = EmbeddingWithPadding,
                        first_block_module = TransfBaseBlockWithAbsPosEmbedding,
                        subsequent_block_module = TransfBaseBlock,
                        causal = True,
                        config = config,
                        name = 'mod' )
    init_params = model.init( rngs=key,
                              datamat=jnp.zeros(unaligned_seq.shape,dtype=int),
                              training=False,
                              sow_intermediates=False)
    test_neural_embedding_blocks(model, variable_dict=init_params)
    del config, model, init_params
    
    config = {'num_heads': 2,
              'num_blocks': n,
              'hidden_dim': 12}
    model = TransfSeqEmb( initial_embed_module = EmbeddingWithPadding,
                        first_block_module = RoPETransfBlock,
                        subsequent_block_module = RoPETransfBlock,
                        causal = True,
                        config = config,
                        name = 'mod' )
    init_params = model.init( rngs=key,
                              datamat=jnp.zeros(unaligned_seq.shape,dtype=int),
                              training=False,
                              sow_intermediates=False)
    test_neural_embedding_blocks(model, variable_dict=init_params)
    del config, model, init_params, n
    
    

####################
### Mamba modules  #
####################
from models.sequence_embedders.mamba.embedders import MambaSeqEmb
from models.sequence_embedders.mamba.blocks_fns import UnidirectResidualMambaLayer

for n in [1, 3]:
    config = {'num_blocks': n,
              'hidden_dim': 10,
              'expansion_factor': 2,
              }
    model = MambaSeqEmb( initial_embed_module = EmbeddingWithPadding,
                        first_block_module = UnidirectResidualMambaLayer,
                        subsequent_block_module = UnidirectResidualMambaLayer,
                        causal = True,
                        config = config,
                        name = 'mod' )
    init_params = model.init( rngs=key,
                              datamat=jnp.zeros(unaligned_seq.shape,dtype=int),
                              training=False,
                              sow_intermediates=False)
    test_neural_embedding_blocks(model, variable_dict=init_params)
    del config, model, init_params
del unaligned_seq


###############################################################################
### FINAL OUTPUT LAYERS   #####################################################
###############################################################################
anc = jnp.array( [[1, 3, 4, 2, 0]] )
padding_mask = anc != 0
anc = jnp.repeat( anc[...,None], 10, axis=-1 )
desc = jnp.array( [[1, 4, 3, 2, 0]] )
desc = jnp.repeat( desc[...,None], 10, axis=-1 )

##################
### feedforward  #
##################
def test_final_pred(model, variable_dict):
    whole_seq_result = model.apply( datamat=unaligned_seq,
                                      variables=variable_dict,
                                      training=False,
                                      sow_intermediates=False)
    if type(whole_seq_result) == tuple:
        whole_seq_result = whole_seq_result[1]
        
    per_site_result = []
    for l in range(1,unaligned_seq.shape[1]+1):
        clipped_input = unaligned_seq[:,:l]
        last_site_result = model.apply( datamat=clipped_input,
                                          variables=variable_dict,
                                          training=False,
                                          sow_intermediates=False )
        if type(last_site_result) == tuple:
            last_site_result = last_site_result[1]
        
        last_site_result = last_site_result[:,-1,:][:,None,:]
        per_site_result.append(last_site_result)
    per_site_result = jnp.concatenate( per_site_result, axis=1 )
    
    assert jnp.allclose(per_site_result, whole_seq_result)


from models.feedforward_predict.FeedforwardPredict import FeedforwardPredict
extra_feat = jnp.array( [[4, 1, 1, 5, 0]] )[...,None]
dmat_lst = [anc, desc, extra_feat]

config = {'layer_sizes': [5, 3],
          'normalize_inputs': True}
model = FeedforwardPredict( config = config,
                            name = 'mod' )
init_params = model.init( rngs=key,
                          datamat_lst=dmat_lst,
                          padding_mask=padding_mask,
                          training=False,
                          sow_intermediates=False)

whole_seq_result = model.apply( datamat_lst=dmat_lst,
                                padding_mask = padding_mask,
                                variables=init_params,
                                training=False,
                                sow_intermediates=False )

per_site_result = []
for l in range(1,anc.shape[1]+1):
    clipped_inputs = [m[:,:l,:] for m in dmat_lst]
    
    last_site_result = model.apply( datamat_lst=clipped_inputs,
                                    padding_mask = padding_mask[:,:l],
                                    variables=init_params,
                                    training=False,
                                    sow_intermediates=False )

    last_site_result = last_site_result[:,-1,:][:,None,:]
    per_site_result.append(last_site_result)
per_site_result = jnp.concatenate( per_site_result, axis=1 )

assert jnp.allclose(per_site_result, whole_seq_result)

del model, init_params, whole_seq_result, per_site_result, l, clipped_inputs
del last_site_result, dmat_lst, config



#################
### neural TKF  #
#################
# only check the "all_local" configuration
from models.neural_hmm_predict.initializers import neural_hmm_params_instance

dmat_lst = [anc, desc]
config = {'indel_model_type': 'tkf92',
          'loss_type': 'joint',
          'emission_alphabet_size': 20,
          'exchang_config': {'use_anc_emb': True,
                             'use_desc_emb': True,
                             'layer_sizes': [8,5],
                             'emission_alphabet_size': 20,
                             'unit_norm_rate_matrix': False
                             },
          'equilibr_config': {'use_anc_emb': True,
                              'use_desc_emb': True,
                              'layer_sizes': [8,5],
                              'emission_alphabet_size': 20
                              },
          'indels_config': {'use_anc_emb': True,
                             'use_desc_emb': True,
                             'layer_sizes': [8,5] 
                             }
          }
model, init_params = neural_hmm_params_instance( input_shapes = [anc.shape, desc.shape],
                            dummy_t_array = jnp.array([[0]]),
                            model_init_rngkey = key, 
                            tabulate_file_loc = None,
                            preset_name = 'all_local', 
                            model_config = config )

whole_seq_result_dict = model.apply( datamat_lst=dmat_lst,
                                     padding_mask = padding_mask,
                                     t_array = jnp.array([[1.]]),
                                     variables=init_params,
                                     training=False,
                                     sow_intermediates=False )

keep = ['FPO_logprob_emit_match', 
        'FPO_logprob_emit_indel', 
        'FPO_logprob_transits']
whole_seq_result_dict = {k:v for k,v in whole_seq_result_dict.items() if k in keep}

per_site_result = {k: [] for k in whole_seq_result_dict.keys()}
for l in range(1,anc.shape[1]+1):
    clipped_inputs = [m[:,:l,:] for m in dmat_lst]
    
    last_site_result_dict = model.apply( datamat_lst=clipped_inputs,
                                    padding_mask = padding_mask[:,:l],
                                    t_array = jnp.array([[1.]]),
                                    variables=init_params,
                                    training=False,
                                    sow_intermediates=False )
    
    to_ap = last_site_result_dict['FPO_logprob_emit_match'][:,:,-1,:,:][:,:,None,:,:]
    per_site_result['FPO_logprob_emit_match'].append(to_ap)

    to_ap = last_site_result_dict['FPO_logprob_transits'][:,:,-1,:,:][:,:,None,:,:]
    per_site_result['FPO_logprob_transits'].append(to_ap)
    
    to_ap = last_site_result_dict['FPO_logprob_emit_indel'][:,-1,:][:,None,:]
    per_site_result['FPO_logprob_emit_indel'].append(to_ap)
    
    
# FPO_logprob_emit_match
pred = jnp.concatenate( per_site_result['FPO_logprob_emit_match'],axis=2)
true = whole_seq_result_dict['FPO_logprob_emit_match']
assert jnp.allclose(pred, true)
del pred, true

# FPO_logprob_transits
pred = jnp.concatenate( per_site_result['FPO_logprob_transits'],axis=2)
true = whole_seq_result_dict['FPO_logprob_transits']
assert jnp.allclose(pred, true)
del pred, true

# FPO_logprob_emit_indel
pred = jnp.concatenate( per_site_result['FPO_logprob_emit_indel'],axis=1)
true = whole_seq_result_dict['FPO_logprob_emit_indel']
assert jnp.allclose(pred, true)

print('[PASS] all model blocks are properly causal')