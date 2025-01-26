#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 12:31:52 2025

@author: farismismar
"""

import numpy as np

seed = 42

np_random = np.random.RandomState(seed=seed)
a = np_random.randint(0, 5, size=2)  # a two-dimensional vector

def rotation_matrix(theta):
    # Rotates a vector in 2D space.
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

max_distance = -np.inf
min_distance = np.inf

min_theta = np.nan
max_theta = np.nan
for idx, t in enumerate(np.arange(-180, 181)):
    r_t = rotation_matrix(t * np.pi/180.)
    b = np.linalg.norm(a- a@r_t, ord=2)
    
    print(f'{t} | {b} | ', end='')
    if b < min_distance:
        min_distance = b
        min_theta = t
        print('min.')
    elif b > max_distance:
        max_distance = b
        max_theta = t
        print('max.')
    else:
        print()
        
    if idx % 10 == 0:
        print('-' * 35)

print(f'Distance is maximum when theta is {max_theta} deg.')
print(f'Distance is minimum when theta is {min_theta} deg.')
