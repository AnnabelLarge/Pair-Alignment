#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 19:42:11 2025

@author: annabel_large

Given the two classes I started with, combine them (with chatGPT magic)
"""

class RateMultipliersPerClass(ModuleBase):
    """
    Class-dependent rate multipliers: P(k | c), ρ_{c,k}

    Config:
    --------
    - 'num_mixtures': int (C)
    - 'k_rate_mults': int (K)
    - 'rate_mult_range': tuple (min_val, max_val), default (0.01, 10)
    - 'norm_rate_mults': bool
    """
    config: dict
    name: str

    def setup(self):
        self.C = self.config['num_mixtures']
        self.K = self.config['k_rate_mults']
        self.rate_mult_min_val, self.rate_mult_max_val = self.config.get('rate_mult_range', (0.01, 10))
        self.norm_rate_mults = self.config.get('norm_rate_mults', True)

        self._init_prob_logits()
        self._init_rate_logits()

        self.rate_multiplier_activation = partial(bound_sigmoid,
                                                  min_val=self.rate_mult_min_val,
                                                  max_val=self.rate_mult_max_val)

    def _init_prob_logits(self):
        self.rate_mult_prob_logits = self.param('rate_mult_prob_logits',
                                                nn.initializers.normal(),
                                                (self.C, self.K),
                                                jnp.float32)

    def _init_rate_logits(self):
        self.rate_mult_logits = self.param('rate_mult_logits',
                                           nn.initializers.normal(),
                                           (self.C, self.K),
                                           jnp.float32)

    def _get_norm_factor(self,
                         rate_multiplier: jnp.ndarray,
                         log_rate_mult_probs: jnp.ndarray,
                         log_class_probs: Optional[jnp.ndarray]):
        if log_class_probs is None:
            raise ValueError("log_class_probs must be provided when norm_rate_mults=True.")
        joint = jnp.exp(log_class_probs[:, None] + log_rate_mult_probs)  # (C, K)
        return jnp.sum(joint * rate_multiplier)

    def __call__(self,
                 log_class_probs: Optional[jnp.ndarray] = None,
                 sow_intermediates: bool = False):
        log_rate_mult_probs = nn.log_softmax(self.rate_mult_prob_logits, axis=-1)
        rate_multiplier = self.rate_multiplier_activation(self.rate_mult_logits)

        if sow_intermediates:
            self.sow_histograms_scalars(jnp.exp(log_rate_mult_probs[0]),
                                        f'{self.name}/prob of rate multipliers',
                                        'scalars')
            self.sow_histograms_scalars(rate_multiplier[0],
                                        f'{self.name}/rate multipliers',
                                        'scalars')

        if self.norm_rate_mults:
            norm = self._get_norm_factor(rate_multiplier, log_rate_mult_probs, log_class_probs)
            rate_multiplier = rate_multiplier / norm

        return log_rate_mult_probs, rate_multiplier


class IndpRateMultipliers(RateMultipliersPerClass):
    """
    Class-independent rate multipliers: P(k), ρ_k shared across c
    """

    def _init_prob_logits(self):
        logits = self.param('rate_mult_prob_logits',
                            nn.initializers.normal(),
                            (self.K,),
                            jnp.float32)
        self.rate_mult_prob_logits = jnp.broadcast_to(logits[None, :], (self.C, self.K))

    def _init_rate_logits(self):
        logits = self.param('rate_mult_logits',
                            nn.initializers.normal(),
                            (self.K,),
                            jnp.float32)
        self.rate_mult_logits = jnp.broadcast_to(logits[None, :], (self.C, self.K))

    def _get_norm_factor(self,
                         rate_multiplier: jnp.ndarray,
                         log_rate_mult_probs: jnp.ndarray,
                         log_class_probs: Optional[jnp.ndarray]):
        # Shared across C, so normalize using just one row
        rho = rate_multiplier[0]
        pk = jnp.exp(log_rate_mult_probs[0])
        return jnp.sum(rho * pk)
