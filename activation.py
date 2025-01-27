#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 18:07:39 2025

@author: farismismar
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Enable LaTeX-like fonts
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
       
# Define the functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def step(x):
    return np.where(x >= 0, 1, 0)

# Generate input data
x = np.linspace(-2, 2, 500)

# Compute the outputs
y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_step = step(x)

# Plot the functions
fig = plt.figure(figsize=(10, 6))
plt.plot(x, y_sigmoid, label="$\sigma(x)$", lw=1.5, color="blue")
plt.plot(x, y_relu, label="$\max(0, x)$", lw=1.5, color="green")
plt.plot(x, y_step, label="$u(x)$", lw=1.5, color="red")

# Customize the plot
plt.xlabel("$x$", fontsize=20)
plt.ylabel("$f(x)$", fontsize=20)
plt.axhline(0, color='black', linewidth=0.5, linestyle="--")
plt.axvline(0, color='black', linewidth=0.5, linestyle="--")
plt.legend(fontsize=24)
plt.grid(alpha=0.3)

# Set integer-only ticks using MaxNLocator
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

# Increase font size for tick labels
plt.tick_params(axis='both', which='major', labelsize=24)

plt.tight_layout()

# Show the plot
plt.show()
plt.close(fig)