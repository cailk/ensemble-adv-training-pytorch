# --coding:utf-8--
'''
@author: cailikun
@time: 2019/4/6 上午11:23
'''
import torch
import numpy as np

MAX_ITERATIONS = 1000
ABORT_EARLY = True
INITIAL_CONST = 1e-3
LEARNING_RATE = 5e-3
LARGEST_CONST = 2e+1
TARGETED = True
CONST_FACTOR = 10.0
CONFIDENCE = 0
EPS = 0.3

class Carlini:
    def __init__(self, model, targeted = TARGETED, learning_rate = LEARNING_RATE, max_iterations = MAX_ITERATIONS,
                 abort_early = ABORT_EARLY, initial_const = INITIAL_CONST, largest_const = LARGEST_CONST,
                 const_factor = CONST_FACTOR, confidence = CONFIDENCE, eps = EPS):
        self.model = model

        self.TARGETED = targeted
        self.LEARNING_RATE = LEARNING_RATE
        self.MAX_ITERATIONS = max_iterations
        self.ABORT_EARLY = abort_early
        self.INITIAL_CONST = initial_const
        self.LARGEST_CONST = largest_const
        self.CONST_FACTOR = const_factor
        self.EPS = eps

