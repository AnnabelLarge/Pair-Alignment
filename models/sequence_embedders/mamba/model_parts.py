#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 18:13:13 2024

@author: annabel

ABOUT:
======
Blocks and pieces to use in Mamba state-space models (mostly from 
  Ian's selectssm.py)

using what he says is the fastest implementation, ssm_chunked_scan

"""
import logging
from typing import Any, Callable, Sequence, Union, Tuple
from dataclasses import field
import math
from functools import reduce
import einops

import jax
import jax.numpy as jnp
import flax.linen as nn

from models.BaseClasses import ModuleBase


###############################################################################
### small helper functions ####################################################
###############################################################################
def inverse_softplus(x):
    return x + jnp.log(1 - jnp.exp(-x))

def debug_log(fmt: str, *args, **kwargs):
    jax.debug.callback(
        lambda *args, **kwargs: logging.warning(fmt.format(*args, **kwargs)),
        *args, **kwargs)

def largest_factor_up_to(b,n):
    if n < 2:
        return n
    k = b
    while n % k != 0:
        k -= 1
    return k


###############################################################################
### scan implementation for SSM layer #########################################
###############################################################################
def ssm_chunked_scan (x, 
                      Acoeff, 
                      Bcoeff, 
                      Ccoeff, 
                      dt, 
                      chunk_size: int = None, 
                      n_channel_groups: int = 1):
    """
    SSM scan function from Ian
    
    sizes:
    ------
    x: (B, L, D)
    Acoeff: (D, N)
    Bcoeff: (B, L, D)
    Ccoeff: (B, L, D)
    dt: (B, L, D) or (B, L, 1) ( can assume (B, L, D) and rely on broadcasting)
    """
    B = x.shape[0]
    L = x.shape[1]
    D = x.shape[2]
    N = Acoeff.shape[-1]

    if n_channel_groups is not None:
        K = n_channel_groups
    else:
        K = 1
    if D % K != 0:
        raise ValueError(f"n_channel_groups={n_channel_groups} must divide D={D}")

    if chunk_size is None:
        chunk_size = largest_factor_up_to(int(math.sqrt(K*L)),L)

    if L % chunk_size != 0:
        raise ValueError(f"chunk_size={chunk_size} must divide L={L}")
    n_chunks = L // chunk_size

    # Transpose length & batch dimensions to make the scan over length, and 
    # split into chunks
    # This is a bit inefficient, but taking dynamic slices appears to be worse
    x_chunks = einops.rearrange (x, 'b (c l) (k d) -> c k l b d', c=n_chunks, k=K)
    A_blocks = einops.rearrange (Acoeff, '(k d) n -> k d n', k=K)
    B_chunks = einops.rearrange (Bcoeff, 'b (c l) n -> c l b n', c=n_chunks)
    C_chunks = einops.rearrange (Ccoeff, 'b (c l) n -> c l b n', c=n_chunks)
    dt_chunks = einops.rearrange (dt, 'b (c l) (k d) -> c k l b d', c=n_chunks, k=K)

    # Function to do an associative scan for a single chunk
    # We decorate this with @jax.remat to flag that we are OK with re-performing this scan whenever needed
    @jax.remat
    def scan_chunk (carry, chunk):
        # For the purposes of shape annotation within this code we write D instead of D/K
        g_init, h_init = carry  # (1, B, D, N)  (1, B, D, N)

        x_chunk, A_block, B_chunk, C_chunk, dt_chunk = chunk
        # dA = exp(A*dt) [zero-order hold], dB = B*dt*x [Euler step]
        dA = jnp.exp (jnp.einsum ('dn,lbd->lbdn', A_block, dt_chunk))  # (chunk_size, B, D, N)
        dB = jnp.einsum ('lbn,lbd,lbd->lbdn', B_chunk, x_chunk, dt_chunk)  # (chunk_size, B, D, N)
        # The associative scan is a product of matrices of the form 
        # ((g,h),(0,1)) where g_i=exp(A*dt)x_i and h_i=B*dt*x_i
        # Since matrices of this form are are closed under multiplication, 
        # we can represent all intermediate products in the same way
        @jax.remat
        def associative_scan_fn (l, r):  # l, r, and return value are tuples of the form ((B,D,N), (B,D,N))
            g_l, h_l = l
            g_r, h_r = r
            return tuple((g_l*g_r, g_r*h_l + h_r))
        gs, hs = jax.lax.associative_scan (associative_scan_fn, (dA, dB))  # (chunk_size, B, D, N)  (chunk_size, B, D, N)
        hs = gs * h_init + hs  # Incorporate h_init here so that it is reflected in y_chunk
        # We only need to keep the last state of gs, so we can discard the 
        # rest. Otherwise we would incorporate g_init here, like so:
        # gs = g_init * gs
        y_chunk = jnp.einsum ('lbn,lbdn->lbd', C_chunk, hs)  # (chunk_size, B, D)
        return (gs[-1:,...] * g_init, hs[-1:,...]), y_chunk  # note g_init incorporated here

    # A wrapper that splits the dimensions into K blocks and does the inner 
    # associative scan for each block, re-using B and C (which don't 
    # change across dimensions)
    @jax.remat
    def scan_chunk_mapped (carry, chunk):
        g_init, h_init = carry  # (K,1,B,D/K,N) (K,1,B,D/K,N)
        
        x_chunk, B_chunk, C_chunk, dt_chunk = chunk   # (K,B,L,D/K), (B,L,N), (B,L,N), (K,B,L,D/K)
        @jax.remat
        def scan_chunk_wrapper (block):
            dA_init_block, dB_init_block, x_chunk_block, A_block, dt_chunk_block = block
            return scan_chunk ((dA_init_block, dB_init_block), (x_chunk_block, A_block, B_chunk, C_chunk, dt_chunk_block))
        return jax.lax.map (scan_chunk_wrapper, (g_init, h_init, x_chunk, A_blocks, dt_chunk))

    # Perform the scan over chunks recurrently (with rematerialization as 
    # noted above), with each chunk being an associative scan
    (_A_final, _h_final), y_chunks = jax.lax.scan (scan_chunk_mapped, 
                                                   (jnp.ones((K,1,B,D//K,N)), 
                                                    jnp.zeros((K,1,B,D//K,N))), 
                                                   (x_chunks, B_chunks, C_chunks, dt_chunks) )  # (K, n_chunks, B, D//K)

    return einops.rearrange (y_chunks, 'c k l b d -> b (c l) (k d)')  # (B, L, D)



###############################################################################
### SelectiveSSM layer ########################################################
###############################################################################
class ConvAndSelectiveSSM(ModuleBase):
    """ 
    A variation on MAMBA v1: https://arxiv.org/pdf/2312.00752.pdf 
      - A_coefficient matrix is NOT dependent on input to layer; 
        it's randomly initialized
    
    
    This specifically does the branch AFTER linear projection and
      BEFORE residual add:
        1. convolution
        2. activation
        3. mask padding tokens
        4. ssm_chunked_scan
        5. mask padding tokens again
        6. add residual connection: jnp.einsum ('bld,d->bld', input, Dcoeff)
        
    
    init with:
    ==========
    reverse (bool): used in bidirecitonal mamba
    config (dict): config unpack
    name (str): SSM layer name
    
    
    config contains:
    ================
    things I could play with:
    -------------------------
        - hidden features, N [default=16]: 
                the lower-dimensional embedding for SSM layer
                
        - dt_rank [default="auto"]: 
                size of dt variable; if I change this later, assert D % dt_rank == 0
                
        - dt_proj [default=True]:
                whether or not to automatically learn dt initialization
                
        - ssm_shift_conv_size [default=3]: 
                the kernel size for the initial 1D convolution
                
        - activation [default: 'silu']:
                activation function i.e. the gating mechanism for the SSM layer
        
        - dt_min, dt_max [default: 0.001, 0.1 respectivly]:
                context window min/max
                
            
    things I should not change:
    ---------------------------
        - complement (I think this is only relevant for DNA models)
        
        - chunk_size (let recursive scan automatically determine this)
        
        - n_channel_groups (let recursive scan automatically determine this)
        
        
        
    apply_fn:
    ==========
    inputs for apply_fn
        - x: matrix of size (B, L, E*D)
        - padding_mat: boolean mask to hide padding tokens
        - sow_intermediates: whether or not to record intermediates values 
                             (for tensorboard)
    
    outputs from apply_fn
        - y: matrix of size (B, L, E*D)
    """
    config: dict
    reverse: bool
    name: str
    
    def setup(self):
        # inputs with defaults
        self.ssm_hidden_features = self.config.get("ssm_hidden_features", 16)  # N
        self.dt_rank = self.config.get("dt_rank", "auto") # R
        self.dt_proj = self.config.get("dt_proj", True) # whether to use a linear projection (vs broadcast) to map dt_rank to D
        self.ssm_shift_conv_size = self.config.get("ssm_shift_conv_size", 3) 
        self.dt_min = self.config.get("dt_min", 0.001) # 1/(long-range context length)
        self.dt_max = self.config.get("dt_max", 0.1) # 1/(short-range context length)
        self.padding_idx = self.config.get("seq_padding_idx", 0) 
        
        # !!! inputs to keep as-is
        self.act_type = 'silu'
        self.complement = False  # only checked if reverse is true
        self.chunk_size = None
        self.n_channel_groups = None
        

    @nn.compact
    def __call__(self, x, padding_mat, sow_intermediates: bool):
        # padding positions in x are zeros
        
        ### 1: get dimensions for parameter blocks
        B = x.shape[0]
        L = x.shape[1]
        D = x.shape[2]  # this is actually E*D; E: expansion factor

        N = self.ssm_hidden_features
 
        if self.dt_rank == 'auto':
            dt_rank = math.ceil(D / 16)
        else:
            dt_rank = self.dt_rank

        
        ### 2: flip input along length dimension (in bidirectional)
        if self.reverse:
            x = jnp.flip (x, axis=(-2,-1) if self.complement else -2)
            padding_mat = jnp.flip(padding_mat, axis=(-2, -1) if self.complement else -2)
            

        ### 3: first 1D convolution            
        u = nn.Conv (features=D, 
                     feature_group_count=D, 
                     kernel_size=(self.ssm_shift_conv_size,), 
                     strides=(1,), 
                     padding= 'CAUSAL',
                     use_bias=False, 
                     name=f"shift_conv", 
                     kernel_init=nn.initializers.lecun_normal()) (x)  # (B, L, E*D)
        
        """
        # (output diagnostics)
        if sow_intermediates:
            self.sow_histograms_scalars(mat = u, 
                                        label = f'{self.name}/conv', 
                                        which=['scalars'])
        """
            
            
        ### 4: first activation after convolution
        u = nn.silu(u)

        """
        # (output diagnostics)
        if sow_intermediates:
            self.sow_histograms_scalars(mat = u, 
                                        label = f'{self.name}/u; after conv and activation', 
                                        which=['scalars'])
        """
        
        # mask out padding tokens before passing input to ssm
        u = jnp.multiply(u, padding_mat)



        ### 5: SSM parameter initialization
        # 5.1: Initialize A nonrandomly with evenly spaced eigenvalues; 
        # keep parameterization in log space to guarantee A<0 
        Acoeff = -jnp.exp (self.param ('A_log', 
                                       lambda rng: jnp.log (jnp.repeat (jnp.arange(start=1,stop=N+1,dtype=jnp.float32)[None,:], 
                                                                        D, 
                                                                        axis=0) 
                                                            ) 
                                       ) 
                           )  # (E*D, N)
        
        # 5.2: initialize B and C directly from convolution output u ( i.e. x(t=0) )
        Bcoeff, Ccoeff = jnp.split (nn.Dense (features=2*N, 
                                              name='BC', 
                                              use_bias=True, 
                                              kernel_init=nn.initializers.lecun_normal()) (u), 
                                    2, 
                                    axis=-1)  # both are (B, L, N)
        
        # 5.3: initialize D, the skip connection
        Dcoeff = self.param ('D', 
                             lambda rng: jnp.ones((D,))
                             )  # (E*D)

        # 5.4: initialize delta_t, the time step, from x_0
        dt_bias_init = lambda rng, shape, dtype: inverse_softplus (jax.random.uniform (rng, 
                                                                                       shape=shape, 
                                                                                       dtype=dtype, 
                                                                                       minval=self.dt_min, 
                                                                                       maxval=self.dt_max) )
        dt = nn.Dense (features=dt_rank, 
                       use_bias=True, 
                       name='dt',
                       kernel_init=nn.initializers.lecun_normal(),
                       bias_init=nn.initializers.zeros if self.dt_proj else dt_bias_init) (u)  # (B, L, dt_rank)
        
        """
        # (output diagnostics)
        if sow_intermediates:
            self.sow_histograms_scalars(mat = dt, 
                                        label = f'{self.name}/dt_lowrank', 
                                        which=['scalars'])
        """
            
        # after linear layer, get final dt; could have this be learnable, if desired
        if self.dt_proj:
            dt = nn.Dense (features=D, 
                           use_bias=True, 
                           kernel_init=nn.initializers.lecun_normal(), 
                           bias_init=dt_bias_init, 
                           name='dt_proj') (dt)  # (B, L, E*D)
        else:
            if dt_rank > 1:  # if dt_rank is 1, we can just rely on broadcasting, and save memory
                if D % dt_rank != 0:
                    raise ValueError(f"dt_rank={dt_rank} must divide D={D}")
                dt = jnp.repeat (dt, D // dt_rank, axis=-1)  # (B, L, E*D)
        dt = nn.activation.softplus (dt)  # (B, L, E*D) 

        """
        # (output diagnostics)
        if sow_intermediates:
            self.sow_histograms_scalars(mat = dt, 
                                        label = f'{self.name}/dt; after softplus', 
                                        which=['scalars'])
            
            self.sow_histograms_scalars(mat = Acoeff, 
                                        label = f'{self.name}/A; expect A<0', 
                                        which=['scalars'])
            
            self.sow_histograms_scalars(mat = Bcoeff, 
                                        label = f'{self.name}/B', 
                                        which=['scalars'])
            
            self.sow_histograms_scalars(mat = Ccoeff, 
                                        label = f'{self.name}/C', 
                                        which=['scalars'])
        """
            
        ### 6: Perform SSM scan, using scan function above
        y = ssm_chunked_scan (x, Acoeff, Bcoeff, Ccoeff, dt, 
                              chunk_size=self.chunk_size, 
                              n_channel_groups=self.n_channel_groups)  # (B, L, E*D)
        
        # mask out padding tokens after ssm, before using skip connection term, D
        # TODO: does this kill gradients..?
        y = jnp.multiply(y, padding_mat)
        

        ### 7: if you originally flipped the input, then flip it back to 
        ###    original orientation (along seqlen L)
        if self.reverse:
            y = jnp.flip (y, axis=(-2,-1) if self.complement else -2)

        """
        # (output diagnostics)
        if sow_intermediates:
            self.sow_histograms_scalars(mat = y, 
                                        label = f'{self.name}/ssm_residual', 
                                        which=['scalars'])
        """

        ### 8: Add in the skip connection term, D
        y = y + jnp.einsum ('bld,d->bld', x, Dcoeff)
        
        return y
        


###############################################################################
### Mamba v1 ##################################################################
###############################################################################
class UnidirecMambaModule(ModuleBase):
    """
    one Mamba v1 module that slides the convolution and SSM scan in ONE 
      direction
    
        |
        v
       in --------- 
        |         |
        v         v
      linear    linear
      project   project
        UP        UP
        |         |
        v         v
      *CONV*     act
        |         |
        v         |
      *act*       |
        |         |
        v         |
    *SSM_SCAN*    |
        |         | 
        v         |
        ---> x <---
             |
             v
           linear
           project
            DOWN
             |
             v
            out
    
    left branch is "x branch", right branch is "z branch" (the gating)
    
    note: this is NOT the overall residual setup/connection, just the single
      mamba layer (equivalent to one MultHeadDotProductAttention layer)
      
    the layers closed with [*] are done in ConvAndSelectiveSSM module part
      > convolution is CAUSAL

    """
    config: dict
    name: str
    
    def setup(self):
        ### parse the config
        # required
        self.in_alph_size = self.config["in_alph_size"]
        self.expansion_factor = self.config["expansion_factor"] # E
        self.hidden_dim = self.config["hidden_dim"]
        
        # optional
        self.padding_idx = self.config.get("seq_padding_idx", 0)
        
        
        ### !!! inputs to keep as-is
        self.complement = False # not relevant for protein models
        self.act_type = 'silu'
        self.act_fn = nn.silu
        
        
    
    @nn.compact
    def __call__(self, datamat, padding_mat, sow_intermediates: bool):
        ### 1: linear project both branches of the residual add (D -> E*D)
        # (B, L, D) -> (B, L, E*D)
        # wherever datamat is a padding token, x_branch and z_branch will still be zeros
        input_features = datamat.shape[-1]  # D
        ED = math.ceil (self.expansion_factor * input_features)
        x_branch, z_branch  = jnp.split (nn.Dense (features=(2*ED), 
                                                   name=f'{self.name}/proj to SSM', 
                                                   kernel_init=nn.initializers.lecun_normal()) (datamat), 
                                         2, 
                                         axis=-1)
        """
        # (output diagnostics)
        if sow_intermediates:
            self.sow_histograms_scalars(mat = x_branch, 
                                        label = f'{self.name}/x_branch before ConvAndSelectiveSSM', 
                                        which=['scalars'])
        """
        
        # expand padding matrix as well
        padding_mat_expanded = jnp.repeat(padding_mat, 
                                          self.expansion_factor,
                                          -1)
        
        
        ### 2: separate functions on x and z branches        
        # x_branch goes through convolution and selective SSM
        x_branch = ConvAndSelectiveSSM(config = self.config,
                                       reverse = False,
                                       name = f'{self.name}/ConvAndSSM')( x = x_branch, 
                                                                         padding_mat = padding_mat_expanded, 
                                                                         sow_intermediates = sow_intermediates)
        
        # z_branch just gets activated
        z_branch = self.act_fn(z_branch)
        
        """
        # (output diagnostics)
        if sow_intermediates:
            self.sow_histograms_scalars(mat = x_branch, 
                                        label = f'{self.name}/x_branch after ConvAndSelectiveSSM', 
                                        which=['scalars'])
            
            self.sow_histograms_scalars(mat = z_branch, 
                                        label = f'{self.name}/z_branch after {self.act_type}', 
                                        which=['scalars'])
        """
        
        
        ### 3: element-wise multiplicative gating
        # since x_branch has zeros at padding positions, datamat will still
        #   have zeros at padding positions
        datamat = x_branch * z_branch
        
        """
        # (output diagnostics)
        if sow_intermediates:
            self.sow_histograms_scalars(mat = datamat, 
                                        label = f'{self.name}/after multipl. gating', 
                                        which=['scalars'])
        """
        
        ### 4: project back down to original hidden_dim
        # (B, L, E*D) -> (B,L,D)
        datamat = nn.Dense (features=input_features, 
                            name=f'{self.name}/proj from SSM', 
                            kernel_init=nn.initializers.lecun_normal()) (datamat)
        
        return datamat


class BidirecMambaModule(ModuleBase):
    """
    one Mamba v1 module that slides the convolution and SSM scan in BOTH 
      directions
    
        |
        v
       in --------- 
        |         |
        v         v
      linear    linear
      project   project
        UP        UP
        |         |
        v         v
      *CONV*     act
        |         |
        v         |
      *act*       |
        |         |
        v         |
    *SSM_SCAN*    |
        |         | 
        v         |
        ---> x <---
             |
             v
           linear
           project
            DOWN
             |
             v
            out
    
    note: this is NOT the overall residual setup/connection, just the single
      mamba layer (equivalent to one MultHeadDotProductAttention layer)
    
    the layers closed with [*] are done in ConvAndSelectiveSSM module part
      > convolutions are STILL CAUSAL in both directions

    """
    config: dict
    name: str
    
    def setup(self):
        ### parse the config
        # required
        self.in_alph_size = self.config["in_alph_size"]
        self.expansion_factor = self.config["expansion_factor"] # E
        self.hidden_dim = self.config["hidden_dim"]
        
        # optional
        self.padding_idx = self.config.get("seq_padding_idx", 0)
        self.tie_in_proj = self.config.get("tie_in_proj", False)
        self.tie_gate = self.config.get("tie_gate", False)
        
        # !!! keep as-is
        self.act_type = 'silu'
        self.act_fn = nn.silu
        self.merge_how = 'concat'
        self.merge_fn = lambda a, b: jnp.concatenate([a, b], axis=-1)
        self.complement = False # not relevant for protein models
        
        # for merge_how=add: self.merge_fn = lambda a, b: jnp.add(a, b)
        
        
    @nn.compact
    def __call__(self, 
                 datamat, 
                 padding_mat, 
                 sow_intermediates: bool):
        ### 1: linear project both branches of the residual add (D -> E*D)
        # take care of both forward and reverse direction in one go
        input_features = datamat.shape[-1]  # D
        n_in_proj = 1 if self.tie_in_proj else 2
        n_gate = 1 if self.tie_gate else 2
        ED = math.ceil (self.expansion_factor * input_features)
        [xf, _xr, zf, _zr] = jnp.split (nn.Dense (features=( (n_in_proj + n_gate) * ED ), 
                                                  name=f'{self.name}/proj to SSM', 
                                                  kernel_init=nn.initializers.lecun_normal()) (datamat), 
                                        [k*ED for k in [1,n_in_proj,n_in_proj+1]], 
                                        axis=-1)
        
        forw_x_branch = xf
        forw_z_branch = zf
        rev_x_branch = xf if self.tie_in_proj else _xr
        rev_z_branch = zf if self.tie_gate else _zr
        del xf, _xr, zf, _zr
        
        # expand padding matrix too
        padding_mat_expanded = jnp.repeat(padding_mat, 
                                          self.expansion_factor,
                                          -1)
        
        """
        # (output diagnostics)
        if sow_intermediates:
            self.sow_histograms_scalars(mat = forw_x_branch, 
                                        label = f'{self.name}/FORW x_branch before ConvAndSelectiveSSM', 
                                        which=['scalars'])
            
            self.sow_histograms_scalars(mat = rev_x_branch, 
                                        label = f'{self.name}/REV x_branch before ConvAndSelectiveSSM', 
                                        which=['scalars'])
        """
        
        ### 2: separate functions on x and z branches
        # both x_branch-es goes through convolution and selective SSM
        # TODO: still using causal convolutions for both directions... does 
        # that make sense...?
        forw_x_branch = ConvAndSelectiveSSM(config = self.config,
                                            reverse = False,
                                            name = f'{self.name}/FW_ConvAndSSM')(x = forw_x_branch,
                                                                                 padding_mat = padding_mat_expanded,
                                                                                 sow_intermediates = sow_intermediates)
        rev_x_branch = ConvAndSelectiveSSM(config = self.config,
                                           reverse = True,
                                           name = f'{self.name}/RV_ConvAndSSM')(x = rev_x_branch,
                                                                                padding_mat = padding_mat_expanded,
                                                                                sow_intermediates = sow_intermediates)
        
        # both z_branch-es just get activated
        forw_z_branch = self.act_fn(forw_z_branch)
        rev_z_branch = self.act_fn(rev_z_branch)
        """
        # (output diagnostics)
        if sow_intermediates:
            mats_to_write = [forw_x_branch, 
                             forw_z_branch, 
                             forw_x_branch, 
                             forw_z_branch]
            suffix_lst = ['FORW x_branch after ConvAndSelectiveSSM',
                          'FORW z_branch after {self.act_type}',
                          'REV x_branch after ConvAndSelectiveSSM',
                          'REV z_branch after {self.act_type}']
            
            for i in range(4):
                self.sow_histograms_scalars(mat = mats_to_write[i], 
                                            label = f'{self.name}/{suffix_lst[i]}', 
                                            which=['scalars'])
        """
        
        ### 3: element-wise multiplicative gating; combine inputs
        forward_x = forw_x_branch * forw_z_branch
        reverse_x = rev_x_branch * rev_z_branch
        datamat = self.merge_fn( forward_x, reverse_x )
        """
        # (output diagnostics)
        if sow_intermediates:
            self.sow_histograms_scalars(mat = datamat, 
                                        label = f'{self.name}/after multipl. gating, merging', 
                                        which=['scalars'])
        """
        
        ### 4: project back down to original hidden_dim
        # (B, L, E*D) -> (B,L,D)
        datamat = nn.Dense (features=input_features, 
                            name=f'{self.name}/proj from SSM', 
                            kernel_init=nn.initializers.lecun_normal()) (datamat)
        
        return datamat
