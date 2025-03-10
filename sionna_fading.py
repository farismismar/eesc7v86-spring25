#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 22:43:25 2025

@author: farismismar
"""

import tensorflow as tf
import numpy as np

import sionna
from sionna.utils import BinarySource, compute_ser, compute_ber
from sionna.channel import FlatFadingChannel
from sionna.mimo import lmmse_equalizer
from sionna.mapping import SymbolDemapper, Mapper, Demapper, Constellation

# Reproducibility
sionna.config.seed = 42

num_tx_ant = 4
num_rx_ant = 4
num_bits_per_symbol = 2
batch_size = 1
codeword_length = 1024  # bits
SNR_dB = 20.  # Transmit SNR [dB]

print(f'Transmit SNR: {SNR_dB:.2f} dB.')

constellation = Constellation("qam", num_bits_per_symbol)
constellation.show()

# Generation of payload (bits)
binary_source = BinarySource()
b = binary_source([batch_size, num_tx_ant, codeword_length])
mapper = Mapper(constellation=constellation)

# Conversion of bits to symbols
x = mapper(b)
shape = tf.shape(x)

# Find the average energy of the transmitted signal x
average_signal_energy = tf.reduce_mean(tf.abs(x) ** 2)
noise_power = average_signal_energy / 10 ** (SNR_dB / 10.)

# Channel effect
channel = FlatFadingChannel(num_tx_ant, num_rx_ant, add_awgn=True, return_channel=True)
x = tf.reshape(x, [-1, num_tx_ant])  # This is necessary for the channel to work.
y, H = channel([x, noise_power])

# Equalization
# The noise correlation matrix is noise_variance times I
rnn = noise_power.numpy() * tf.eye(num_rx_ant, num_rx_ant, dtype=y.dtype)
x_hat, noise_power_effective = lmmse_equalizer(y, H, s=rnn)

# Symbol detection
symbol_demapper = SymbolDemapper("qam", num_bits_per_symbol, hard_out=True)
x_ind = symbol_demapper([x, noise_power])
x_hat_ind = symbol_demapper([x_hat, noise_power_effective])
ser = compute_ser(x_ind, x_hat_ind)
print("SER: {:.4f}".format(ser))

# Conversion to bits
demapper = Demapper("app", constellation=constellation)
llr = demapper([x_hat, noise_power_effective])  # log likelihood ratios
llr = tf.reshape(llr, [batch_size, num_tx_ant, codeword_length])
b_hat = tf.cast(tf.less(0., llr), dtype=tf.float32)

ber = compute_ber(b, b_hat)
print("BER: {:.4f}".format(ber))