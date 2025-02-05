#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 19:01:01 2025

@author: annabel

fake alignments:

    <b> A - D <e>
    <b> C C - <e>
    
    (1, 0)
    (2, 1)
    (2, 2)
    (3, 2)

"""
import numpy as np


aligned_input = np.array([[1,  3, 43,  5,  2],
                          [1,  4,  4, 43,  2],
                          [1,  2,  2,  3, -9],
                          [0,  1,  2,  2, -9]]).T[None,...]

unaligned_seqs = np.array([[1, 3, 5, 2, 0],
                           [1, 4, 4, 2, 0]]).T[None,...]

with open('toyAlign_aligned_mats.npy','wb') as g:
    np.save(g, aligned_input)

with open('toyAlign_seqs_unaligned.npy','wb') as g:
    np.save(g, unaligned_seqs)



out = np.zeros( (20,) )
out[0] += 1
out[1] += 2
out[2] += 1


with open('toyAlign_AAcounts.npy','wb') as g:
    np.save(g, out)


