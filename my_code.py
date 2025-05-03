#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 16:14:38 2025

@author: annabel_large
"""
def myfunc(x):
    return x + 10

class MyClass:
    def __call__(self, x):
        return myfunc(x)

