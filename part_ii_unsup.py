#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:45:37 2025

@author: farismismar
"""

import numpy as np

size = 1000
tx_snr = 10 ** (20/10.)  # 20 dB

seed = 42
random_state = np.random.RandomState(seed=seed)

qpsk_alphabet = np.array([1+1j, 1-1j, -1-1j, -1+1j])

# Normalize qpsk_alphbabet
qpsk_alphabet /= np.sqrt(2.)  # why?

transmit_symbol_idx = random_state.choice(range(len(qpsk_alphabet)), size=size, replace=True)
transmit_symbols = qpsk_alphabet[transmit_symbol_idx]

symbol_power = np.mean(np.abs(transmit_symbols) ** 2)
noise_power = symbol_power / tx_snr

noise = np.sqrt(noise_power / 2.) * (random_state.normal(loc=0, scale=1, size=size) + 1j * \
    random_state.normal(loc=0, scale=1, size=size))

# noise_power is np.var(noise)
received_symbols = transmit_symbols + noise

# Plot the symbols
import matplotlib.pyplot as plt

fig = plt.figure()
plt.scatter(np.real(received_symbols), np.imag(received_symbols), c='r', s=4)
plt.scatter(np.real(qpsk_alphabet), np.imag(qpsk_alphabet), c='k', s=4)

# Axes
plt.axhline(0, color='black')
plt.axvline(0, color='black')

plt.xlabel('I')
plt.ylabel('Q')
plt.grid(True)
plt.show()
plt.close(fig)

from sklearn.cluster import KMeans
centroids = np.c_[np.real(qpsk_alphabet), np.imag(qpsk_alphabet)]
kmeans = KMeans(n_clusters=4, init=centroids, n_init=1, random_state=random_state).fit(centroids)

received_symbols_real = np.c_[np.real(received_symbols), np.imag(received_symbols)]
detected_symbols_real_idx = kmeans.predict(received_symbols_real)

fig = plt.figure()
plt.scatter(np.real(received_symbols), np.imag(received_symbols), cmap='jet', c=detected_symbols_real_idx, s=4)
plt.scatter(np.real(qpsk_alphabet), np.imag(qpsk_alphabet), c='k', s=4)

# Axes
plt.axhline(0, color='black')
plt.axvline(0, color='black')

plt.xlabel('I')
plt.ylabel('Q')
plt.grid(True)
plt.show()
plt.close(fig)

# Compute error
error = np.mean(transmit_symbol_idx != detected_symbols_real_idx)

print(f'k-means symbol detection error is: {error:.4f}.')