"""
From Ian; implements MAMBA in flax

"""

import logging
from typing import Any, Callable, Sequence, Union, Tuple
from dataclasses import field

import math
from functools import reduce

import einops
import flax.linen as nn

import jax
import jax.numpy as jnp

from ssmrecscan import ssm_recursive_scan, ssm_scan


######################
### helper functions #
######################
def l2_norm(params, alpha = 1.):
    return alpha * jnp.sum (jnp.array ([jnp.sum(x*x) for x in jax.tree_util.tree_leaves(params)]))

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


#####################################
### A third scan method             #
### I think this one is the fastest #
#####################################
# x: (B, L, D)
# Acoeff: (D, N)
# Bcoeff: (B, L, N)
# Ccoeff: (B, L, N)
# dt: (B, L, D) or (B, L, 1);  can assume (B, L, D) and rely on broadcasting
def ssm_chunked_scan (x, Acoeff, Bcoeff, Ccoeff, dt, chunk_size: int = None, n_channel_groups: int = 1):
    B = x.shape[-3]
    L = x.shape[-2]
    D = x.shape[-1]
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



######################
### the module class #
######################
class SelectiveSSM(nn.Module):
    """ 
    A variation on MAMBA: https://arxiv.org/pdf/2312.00752.pdf 
    
    init with:
    ----------
    things I could play with
        - reverse [default=False]: 
                used in bidirectional mamba
                
        - hidden features, N [default=16]: 
                the lower-dimensional embedding for SSM layer
                
        - dt_rank [default="auto"]: 
                size of dt variable; if I change this later, assert D % dt_rank == 0
                
        - dt_proj [default=True]:
                whether or not to automatically learn dt initialization
                
        - l2_scale [default=0.0]: 
                l2 penalty
                
        - shift_conv_size [default=3]: 
                the kernel size for the initial 1D convolution
                
        - activation [default: 'silu']:
                activation function i.e. the gating mechanism for the SSM layer
            
    things I should not change
        - complement (I think this is only relevant for DNA models)
        - chunk_size (let recursive scan automatically determine this)
        - n_channel_groups (let recursive scan automatically determine this)
        - dt_min, dt_max (these were originally defined in mamba paper and 
                are probably find to keep)
        - diagnostics (dictionary of diagnostics collected during the run)
        - recursive_scan, custom_vjp_scan (alternative ways of doing SSM scan; 
                Ian recommends I don't use these)
          > min_recursion_length, recursive_split only used in these 
                alternative methods
        
        
    apply_fn:
    ---------
    inputs for apply_fn
        - x: matrix of size (B, L, D)
        - training: if model is in training mode or not (only matters for 
                    recording diagnostics)
    
    outputs from apply_fn
        - y: matrix of size (B, L, D)
    """
    ### inputs to vary in hyperparam sweeps
    reverse: bool = False
    hidden_features: int = 16  # N
    dt_rank: Union[int, str] = 'auto'  # R
    dt_proj: bool = True   # whether to use a linear projection (vs broadcast) to map dt_rank to D
    l2_scale: float = 0.0
    shift_conv_size: int = 3
    activation: str = "silu"
    
    
    ### inputs to keep as-is
    complement: bool = False  # only checked if reverse is true
    chunk_size: int = None
    n_channel_groups: int = None
    dt_min: float = 0.001  # 1/(long-range context length)
    dt_max: float = 0.1    # 1/(short-range context length)
    diagnostics: dict = field(default_factory=dict)
    recursive_scan: bool = False
    custom_vjp_scan: bool = False
    min_recursion_length: int = 2
    recursive_split: int = 2
    #a_init_scale: float = 1.0 (not used?)


    @nn.compact
    def __call__(
        self,
        x,  # (B, L, D)
        train: bool = False,
    ):
        
        ### 1: get dimensions for parameter blocks
        B = x.shape[-3]
        L = x.shape[-2]
        D = x.shape[-1]  # if called by BidirectionalMamba, this is actually E*D
        # E: expansion factor

        N = self.hidden_features
 
        if self.dt_rank == 'auto':
            dt_rank = math.ceil(D / 16)
        else:
            dt_rank = self.dt_rank

        # (output diagnostics)
        if train and 'ssm_input_norm' in self.diagnostics:
            self.sow("diagnostics", "ssm_input_mean", jnp.mean(x))
            self.sow("diagnostics", "ssm_input_sd", jnp.std(x))


        ### 2: flip input along length dimension (in bidirectional)
        if self.reverse:
            x = jnp.flip (x, axis=(-2,-1) if self.complement else -2)


        ### 3: first 1D convolution
        u = nn.Conv (features=D, 
                     feature_group_count=D, 
                     kernel_size=(self.shift_conv_size,), 
                     strides=(1,), 
                     padding="SAME", 
                     use_bias=False, 
                     name="shift_conv", 
                     kernel_init=nn.initializers.lecun_normal()) (x)  # (B, L, D)

        # (output diagnostics)
        if train and 'ssm_coeffs' in self.diagnostics:
            self.sow("diagnostics", "conv_mean", jnp.mean(u))
            self.sow("diagnostics", "conv_sd", jnp.std(u))


        ### 4: first activation after convolution
        if self.activation == "gelu":
            u = nn.gelu(u)
        elif self.activation == "relu":
            u = nn.relu(u)
        elif self.activation == "silu":
            u = nn.silu(u)
        elif self.activation is not None:
            raise Exception(f"Unknown activation: {self.activation}")


        ### 5: SSM parameter initialization
        # 5.1: Initialize A nonrandomly with evenly spaced eigenvalues; 
        # keep parameterization in log space to guarantee A<0 (ask Ian about this)
        Acoeff = -jnp.exp (self.param ('A_log', lambda rng: jnp.log (jnp.repeat (jnp.arange(start=1,stop=N+1,dtype=jnp.float32)[None,:], 
                                                                                 D, 
                                                                                 axis=0)) ) )  # (D, N)
        
        # 5.2: initialize B and C directly from convolution output u ( i.e. x(t=0) )
        Bcoeff, Ccoeff = jnp.split (nn.Dense (features=2*N, 
                                              name='BC', 
                                              use_bias=True, 
                                              kernel_init=nn.initializers.lecun_normal()) (u), 
                                    2, 
                                    axis=-1)  # (B, L, N) *2
        
        # 5.3: initialize D, the skip connection
        Dcoeff = self.param ('D', lambda rng: jnp.ones((D,)))  # (D,)

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
        
        # (output diagnostics)
        if train and 'ssm_coeffs' in self.diagnostics:
            self.sow("diagnostics", "dt_lowrank_mean", jnp.mean(dt))
            self.sow("diagnostics", "dt_lowrank_sd", jnp.std(dt))

        # after linear layer, get final dt; could have this be learnable, if desired
        if self.dt_proj:
            dt = nn.Dense (features=D, 
                           use_bias=True, 
                           kernel_init=nn.initializers.lecun_normal(), 
                           bias_init=dt_bias_init, 
                           name='dt_proj') (dt)  # (B, L, D)
        else:
            if dt_rank > 1:  # if dt_rank is 1, we can just rely on broadcasting, and save memory
                if D % dt_rank != 0:
                    raise ValueError(f"dt_rank={dt_rank} must divide D={D}")
                dt = jnp.repeat (dt, D // dt_rank, axis=-1)  # (B, L, D)
        dt = nn.activation.softplus (dt)  # (B, L, D) or (B, L, 1)

        # (output diagnostics)
        if train and 'ssm_coeffs' in self.diagnostics:
            self.sow("diagnostics", "activated_conv_mean", jnp.mean(u))
            self.sow("diagnostics", "activated_conv_sd", jnp.std(u))
            self.sow("diagnostics", "dt_mean", jnp.mean(dt))
            self.sow("diagnostics", "dt_sd", jnp.std(dt))
            self.sow("diagnostics", "A_mean", jnp.mean(Acoeff))
            self.sow("diagnostics", "A_sd", jnp.std(Acoeff))
            self.sow("diagnostics", "B_sd", jnp.std(Bcoeff))
            self.sow("diagnostics", "C_sd", jnp.std(Ccoeff))


        ### 6: Perform SSM scan 
        # option 1: fully define custom forward and backwards methods
        if self.custom_vjp_scan:
            y = ssm_scan (x, Acoeff, Bcoeff, Ccoeff, dt, 
                          min_recursion_length=self.min_recursion_length, 
                          recursive_split=self.recursive_split)  # (B, L, D)
        
        # option 2: custon forward function; default backwards method from autodiff engine
        elif self.recursive_scan:
            y = ssm_recursive_scan (x, Acoeff, Bcoeff, Ccoeff, dt, 
                                    min_recursion_length=self.min_recursion_length, 
                                    recursive_split=self.recursive_split)  # (B, L, D)
        
        # DEFAULT: the chunked_scan from above
        else:
            y = ssm_chunked_scan (x, Acoeff, Bcoeff, Ccoeff, dt, 
                                  chunk_size=self.chunk_size, 
                                  n_channel_groups=self.n_channel_groups)  # (B, L, D)


        ### 7: if you originally flipped the input, then flip it back to 
        ###    original orientation (along seqlen L)
        if self.reverse:
            y = jnp.flip (y, axis=(-2,-1) if self.complement else -2)

        # (output diagnostics)
        if train and 'ssm_residual' in self.diagnostics:
            self.sow("diagnostics", "ssm_residual_mean", jnp.mean(y))
            self.sow("diagnostics", "ssm_residual_sd", jnp.std(y))


        ### 8: Add in the skip connection term, D
        y = y + jnp.einsum ('bld,d->bld', x, Dcoeff)


        ### 9: Regularizers
        if train:
            # add l2 norm for params
            self.sow("losses", "ssm_regularizer", l2_norm (self.variables['params'], self.l2_scale))

        # (output diagnostics)
        if train and 'ssm_output_norm' in self.diagnostics:
            self.sow("diagnostics", "ssm_output_mean", jnp.mean(y))
            self.sow("diagnostics", "ssm_output_sd", jnp.std(y))

        return y


###########################
### Bidirectional version #
###########################
class BidirectionalMamba(nn.Module):
    """
    Run SelectiveSSM in both directions, then concatenate or add results. Also 
    has an optional MLP at the end, to mimic a transformer block
    
    init with:
    ----------
    things I could play with, unique to this class
        - expansion_factor, E [float, NO DEFAULT]:
                project to E*D dimension before SSM (should be larger than D?)
        
        - concatenate_fwd_rev [default=True]
                if true, concatenate representations; if false, add them        
        
        - norm type [default='rms']
                normalization before/after projection
        
        - bn_momentum [default=0.9]
                only relevant if you batch norm as your norm type
        
        - mlp_layer [default=False]
                whether or not to mimic a transformer, and follow the SSM layer 
                with a small MLP
        
        - dense expansion [default=2]
                used to determine internal size of mlp_layer
        
        - mlp_dropout_rate [default=0.1]
                dropout rate used in mlp_layer
        
        - l2_scale [default=1e=6]: 
                l2 penalty (note that this default is different from 
                SelectiveSSM default... I think it would be possible 
                to use two different l2 regularizers here)
                
        - ssm_args
                this dictionary contains params I pass onto SelectiveSSM
                
    
    things I could play with, also used as arguments for SelectiveSSM
        - hidden features, N [default=16]: 
                the lower-dimensional embedding for SSM layer
        
        - dt_rank [default="auto"]: 
                size of dt variable; if I change this later, assert D % dt_rank == 0
        
        - activation [default: 'silu']:
                activation function i.e. the gating mechanism for the SSM layer
         
            
    things I should not change
        - complement, tie_in_proj, tie_gate (I think this is only relevant 
                for DNA models)
        - diagnostics (dictionary of diagnostics collected during the run)
        
    
    apply_fn:
    ---------
    inputs for apply_fn
        - x: matrix of size (B, L, D)
        - training: if model is in training mode or not (only matters for 
                    recording diagnostics)
    
    outputs from apply_fn
        - y: matrix of size (B, L, D)
    """
    ### inputs to vary in hyperparam sweeps, unique to this class
    expansion_factor: float  # E
    concatenate_fwd_rev: bool = True
    norm_type: str = "rms"
    bn_momentum: float = 0.9
    mlp_layer: bool = False
    dense_expansion: int = 2
    mlp_dropout_rate: float = 0.1
    
    
    ### inputs to vary in hyperparam sweeps, shared with/input into SelectiveSSM
    ssm_args: dict = field(default_factory=dict)
    l2_scale: float = 1e-6
    hidden_features: int = 16   # N
    dt_rank: Union[int, str] = 'auto'
    activation: str = "silu"
    
    
    ### inputs to keep as-is
    # For an RC-equivariant model, set all of 
    # {complement,tie_in_proj,tie_gate,concatenate_fwd_rev} to True
    complement: bool = False
    tie_in_proj: bool = False
    tie_gate: bool = False
    diagnostics: dict = field(default_factory=dict)


    @nn.compact
    def __call__(self, 
                 x, # (B, L, D)
                 train: bool = False):

        ### 1: get dimensions for parameter blocks
        input_features = x.shape[-1]  # D
        
        if self.dt_rank == 'auto':
            dt_rank = math.ceil(input_features / 16)
        else:
            dt_rank = self.dt_rank


        ### 2: choose an activation function to use later
        if self.activation == "gelu":
            activate = nn.gelu
        elif self.activation == "silu":
            activate = nn.silu
        elif self.activation == "relu":
            activate = nn.relu
        else:
            raise Exception(f"Unknown activation: {self.activation}")


        ### 3: save initial input for later skip connection
        skip = x
        
        # (output diagnostics)
        if 'skip' in self.diagnostics and train:
            self.sow ("diagnostics", "skip_mean", jnp.mean(skip))
            self.sow ("diagnostics", "skip_sd", jnp.std(skip))


        ### 4: normalize input; default is RMS norm
        if self.norm_type == "batch":
            x = nn.BatchNorm(momentum=self.bn_momentum, use_running_average=not train)(x)
        elif self.norm_type == "layer":
            x = nn.LayerNorm()(x)
        elif self.norm_type == "group":
            x = nn.GroupNorm()(x)
        elif self.norm_type == "rms":
            x = nn.RMSNorm()(x)


        ### 5: project to expanded dimension (take care of both directions 
        ###    in one dense layer)
        ED = math.ceil (self.expansion_factor * input_features)
        n_in_proj = 1 if self.tie_in_proj else 2
        n_gate = 1 if self.tie_gate else 2
        [xf, _xr, zf, _zr] = jnp.split (nn.Dense (features=((n_in_proj+n_gate)*ED), 
                                                  name='in_proj', 
                                                  kernel_init=nn.initializers.lecun_normal()) (x), 
                                        [k*ED for k in [1,n_in_proj,n_in_proj+1]], 
                                        axis=-1)
        xr = xf if self.tie_in_proj else _xr
        zr = zf if self.tie_gate else _zr


        ### 6: run forward and backward SSM
        xf = SelectiveSSM(hidden_features=self.hidden_features, 
                          reverse=False, 
                          dt_rank=dt_rank, 
                          diagnostics=self.diagnostics, 
                          **self.ssm_args) (xf, train)
        
        xr = SelectiveSSM(hidden_features=self.hidden_features, 
                          reverse=True, 
                          complement=self.complement, 
                          dt_rank=dt_rank, 
                          diagnostics=self.diagnostics, 
                          **self.ssm_args) (xr, train)

        # (output diagnostics)
        if 'gate' in self.diagnostics and train:
            self.sow ("diagnostics", "gate_fwd_mean", jnp.mean(zf))
            self.sow ("diagnostics", "gate_fwd_sd", jnp.std(zf))
            self.sow ("diagnostics", "gate_rev_mean", jnp.mean(zr))
            self.sow ("diagnostics", "gate_rev_sd", jnp.std(zr))

        # concatenate (or add) forward and backward channels, multiplied by 
        # respective activated gates
        if self.concatenate_fwd_rev:
            x = jnp.concatenate ([xf * activate(zf), xr * activate(zr)], axis=-1)
        else:
            x = xf * activate(zf) + xr * activate(zr)

        # (output diagnostics)
        if 'gated' in self.diagnostics and train:
            self.sow ("diagnostics", "gated_mean", jnp.mean(x))
            self.sow ("diagnostics", "gated_sd", jnp.std(x))


        ### 7: project back down to original (B,L,D)
        x = nn.Dense (features=input_features, 
                      name='out_proj', 
                      kernel_init=nn.initializers.lecun_normal()) (x)


        ### 8: residual add
        # (output diagnostics)
        if 'residual' in self.diagnostics and train:
            self.sow ("diagnostics", "residual_mean", jnp.mean(x))
            self.sow ("diagnostics", "residual_sd", jnp.std(x))

        x = skip + x


        ### 9: MLP layer (optional; mirrors the transformer architecture)
        if self.mlp_layer:
            skip = x
            x = nn.Dense(self.dense_expansion*input_features, 
                         name="mlp", 
                         kernel_init=nn.initializers.lecun_normal())(x)
            
            x = nn.Dropout(rate=self.mlp_dropout_rate, deterministic=not train)(x)
            
            x = activate(x)
            
            x = nn.Dense(input_features, 
                         name="mlp_proj", 
                         kernel_init=nn.initializers.lecun_normal())(x)
            
            x = nn.Dropout(rate=self.mlp_dropout_rate, deterministic=not train)(x)
            
            x = skip + x


        ### 10: Regularizers
        if train:
            self.sow("losses", "mamba_regularizer", l2_norm (self.variables['params'], self.l2_scale))

        return x
