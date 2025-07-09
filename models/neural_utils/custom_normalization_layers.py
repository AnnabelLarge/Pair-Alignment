#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 16:45:32 2025

@author: annabel
"""
from flax import linen as nn
import jax.numpy as jnp


class LayerNormOverLastTwoDims(nn.Module):
    epsilon: float = 1e-5
    use_scale: bool = True
    use_bias: bool = True

    def setup(self):
        self.scale = self.param('scale', nn.initializers.ones, (1, 1))
        self.bias = self.param('bias', nn.initializers.zeros, (1, 1))

    def __call__(self, x, mask=None):
        """
        x: shape (..., L, H) — normalize over L and H jointly, per instance
        mask: shape (..., L) = include, 0 = ignore
        """
        B, L, C = x.shape

        if mask is not None:
            # Expand to match x
            mask_exp = jnp.broadcast_to(mask[...,None], x.shape)  # (B, L, H)
            mask_bc = jnp.broadcast_to(mask_exp, x.shape)

            # Zero out masked entries for stats
            x_masked = jnp.where(mask_bc, x, 0.0)

            # Count of unmasked elements per instance (B,)
            count = jnp.sum(mask_bc, axis=(-1, -2), keepdims=True)
            mean = jnp.sum(x_masked, axis=(-1, -2), keepdims=True) / (count + self.epsilon)

            # E[X^2] - (E[X])^2 trick for variance
            x2_masked = jnp.where(mask_bc, x**2, 0.0)
            mean_sq = mean**2
            E_x2 = jnp.sum(x2_masked, axis=(-1, -2), keepdims=True) / (count + self.epsilon)
            var = E_x2 - mean_sq

        else:
            # Unmasked case: normalize over (L, H) for each B
            mean = jnp.mean(x, axis=(-1, -2), keepdims=True)
            var = jnp.var(x, axis=(-1, -2), keepdims=True)
        
        normed = (x - mean) / jnp.sqrt(var + self.epsilon)

        # Apply learnable scale and bias
        if self.use_scale:
            normed = normed * self.scale
        if self.use_bias:
            normed = normed + self.bias
        
        # Zero out masked positions again, if mask was provided
        if mask is not None:
            normed = jnp.where(mask_exp, normed, 0.0)

        return normed




# class InstanceNormOverLastDim(nn.Module):
#     epsilon: float = 1e-5
#     use_scale: bool = True
#     use_bias: bool = True

#     @nn.compact
#     def __call__(self, x, mask=None):
#         """
#         x: (B, L, H) — normalize over the last axis (channel/feature)
#         mask: (B, L) or broadcastable to x — 1 = keep, 0 = ignore (padding)
#         """
#         if mask is not None:
#             # Broadcast mask to match x
#             mask_exp = jnp.broadcast_to(mask[...,None], x.shape)  # (B, L, H)
#             masked_x = jnp.where(mask_exp, x, 0.0) #(B, L, H)
            
#             count = jnp.sum(mask_exp, axis=-1, keepdims=True)
#             mean = jnp.sum(masked_x, axis=-1, keepdims=True) / (count + self.epsilon)

#             # variance = E[x^2] - (E[x])^2
#             mean_sq = mean ** 2
#             sq_x = jnp.where(mask_exp, x**2, 0.0)
#             E_xsq = jnp.sum(sq_x, axis=-1, keepdims=True) / (count + self.epsilon)
#             var = E_xsq - mean_sq
#         else:
#             mean = jnp.mean(x, axis=-1, keepdims=True)
#             var = jnp.var(x, axis=-1, keepdims=True)

#         normed = (x - mean) / jnp.sqrt(var + self.epsilon)

#         if self.use_scale:
#             scale = self.param('scale', nn.initializers.ones, (x.shape[-1],))
#             normed = normed * scale
#         if self.use_bias:
#             bias = self.param('bias', nn.initializers.zeros, (x.shape[-1],))
#             normed = normed + bias

#         # Zero out masked positions again, if mask was provided
#         if mask is not None:
#             normed = jnp.where(mask_exp, normed, 0.0)

#         return (normed, mean, var)
    
# if __name__ == '__main__':
#     import jax
#     import numpy as np
    
#     x1 = np.array( [[1, 2, 3, 0, 0],
#                     [4, 5, 6, 0, 0]]).T[None,...]
    
#     x2 = np.array( [[1, 2, 3, 4, 0],
#                     [4, 5, 6, 7, 0]]).T[None,...]
    
#     x3 = np.array( [[1, 2, 3, 4, 5],
#                     [4, 5, 6, 7, 8]]).T[None,...]
    
#     x = np.concatenate( [x1, x2, x3], axis=0 )
#     del x1, x2, x3
    
#     mask = (x != 0.0)[...,0]
    
#     mymodel = InstanceNormOverLastDim(use_bias = False,
#                                       use_scale = False)
#     init_params = mymodel.init(rngs=jax.random.key(0),
#                                x = jnp.zeros(x.shape),
#                                mask = mask)
    
#     out = mymodel.apply(variables=init_params,
#                         x=x,
#                         mask=mask)
#     normed_with_mask, mean_with_mask, var_with_mask = out
#     del out
    
#     out = mymodel.apply(variables=init_params,
#                         x=x)
#     normed_all, mean_all, var_all = out
#     del out
    