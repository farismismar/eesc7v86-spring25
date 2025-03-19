#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 20:54:45 2025

@author: farismismar
"""

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

np_random = np.random.RandomState(seed=42)

N_sc = 64
N_r = 4
N_t = 4 
frequency = 1800e6
Df = 15e3

def _generate_cdl_a_channel(N_sc, N_r, N_t, carrier_frequency):
    global np_random, Df
    
    # Channel parameters from CDL-A model (delays, powers, AoD, AoA)
    delay_taps = np.array([0.0, 0.3819, 0.4025, 0.5868, 0.4610, 0.5375, 0.6708, 0.5750, 0.7618, 1.5375, 1.8978, 2.2242, 2.1718, 2.4942, 2.5119, 3.0582, 4.0810, 4.4579, 4.5695, 4.7966, 5.0066, 5.3043, 9.6586])
    powers_dB = np.array([-13.4, 0.0, -2.2, -4.0, -6.0, -8.2, -9.9, -10.5, -7.5, -15.9, -6.6, -16.7, -12.4, -15.2, -10.8, -11.3, -12.7, -16.2, -18.3, -18.9, -16.6, -19.9, -29.7])
    aod = np.array([-178.1, -4.2, -4.2, -4.2, 90.2, 90.2, 90.2, 121.5, -81.7, 158.4, -83.0, 134.8, -153.0, -172.0, -129.9, -136.0, 165.4, 148.4, 132.7, -118.6, -154.1, 126.5, -56.2])
    aoa = np.array([51.3, -152.7, -152.7, -152.7, 76.6, 76.6, 76.6, -1.2, 10.5, -45.6, 88.1, 34.5, 45.0, -28.4, -90.2, 12.8, -9.7, 37.4, 28.2, 15.7, 3.0, 5.0, 16.0])

    num_taps = len(powers_dB)
    
    # Initialize MIMO OFDM channel matrix H with dimensions (N_sc, N_r, N_t)
    H = np.zeros((N_sc, N_r, N_t), dtype=np.complex128)

    # Frequency range for the subcarriers
    subcarrier_frequencies = carrier_frequency + (np.arange(N_sc) - N_sc // 2) * Df   # subcarrier indices
    
    # Generate channel response for each tap and apply delay phase shifts
    for tap in range(num_taps):
        # Delay in seconds
        delay = delay_taps[tap] * 1e-6  # convert from microseconds to seconds
        power = 10 ** (powers_dB[tap] / 10.)  # Linear scale of power
        aod_rad = np.radians(aod[tap])
        aoa_rad = np.radians(aoa[tap])
    
        # Apply the phase shift for each subcarrier based on the delay
        phase_shift = np.exp(-2j * np.pi * subcarrier_frequencies * delay)
        
        # For each subcarrier and symbol, calculate the MIMO channel response
        for sc in range(N_sc):
            # Generate the channel matrix for this subcarrier and symbol
            # For each antenna, the channel response is influenced by the AoD and AoA
            # Complex Gaussian fading for each tap, scaled by tap power
            H_tap = np.sqrt(power) * np.outer(np.exp(1j * aod_rad), np.exp(1j * aoa_rad)) * \
                                     (np_random.randn(N_sc, N_r, N_t) + 1j * np_random.randn(N_sc, N_r, N_t)) / np.sqrt(2)
    
            # Apply phase shift across subcarriers
            H += H_tap * phase_shift[sc]
        
    # Normalize channel gains
    for sc in range(N_sc):
        H[sc, :, :] /= np.linalg.norm(H[sc, :, :], ord='fro')
    
    return H


def _generate_cdl_c_channel(N_sc, N_r, N_t, carrier_frequency):
    global np_random, Df

    # Generates a 3GPP 38.900 CDL-C channel with dimensions (N_sc, N_r, N_t).
    delay_taps = [0, 0.209, 0.423, 0.658, 1.18, 1.44, 1.71]  # in microseconds
    powers_dB = [-0.2, -13.5, -15.4, -18.1, -20.0, -22.1, -25.2]  # tap power in dBm

    # Convert dB to linear scale for power
    powers_linear = 10 ** (np.array(powers_dB) / 10)
    num_taps = len(powers_dB)

    # Initialize the channel matrix (complex random Gaussian per subcarrier, tap, and antenna pair)
    H = np.zeros((N_sc, N_r, N_t), dtype=np.complex128)

    # Frequency range for the subcarriers
    subcarrier_frequencies = carrier_frequency + (np.arange(N_sc) - N_sc // 2) * Df   # subcarrier indices

    # Generate channel response for each tap and apply delay phase shifts
    for tap in range(num_taps):
        # Delay in seconds
        delay = delay_taps[tap] * 1e-6  # convert from microseconds to seconds

        # Apply the phase shift for each subcarrier based on the delay
        phase_shift = np.exp(-2j * np.pi * subcarrier_frequencies * delay)  # shape: (N_sc,)

        # Complex Gaussian fading for each tap, scaled by tap power
        H_tap = np.sqrt(powers_linear[tap]) * \
                (np_random.randn(N_r, N_t, N_sc) + 1j * np_random.randn(N_r, N_t, N_sc)) / np.sqrt(2)

        # Apply phase shift across subcarriers
        H += (H_tap * phase_shift).transpose(2, 0, 1)  # Adjust dimensions (N_sc, N_r, N_t)

    # Normalize channel gains
    for sc in range(N_sc):
        H[sc, :, :] /= np.linalg.norm(H[sc, :, :], ord='fro')

    return H


def _generate_cdl_e_channel(N_sc, N_r, N_t, carrier_frequency):
    global np_random, Df

    # Generates a 3GPP 38.900 CDL-E channel with dimensions (N_sc, N_r, N_t).
    delay_taps = [0, 0.264, 0.366, 0.714, 1.53, 1.91, 3.52, 4.20, 5.35]  # in microseconds
    powers_dB = [-0.03, -4.93, -8.03, -10.77, -15.86, -18.63, -21.11, -22.50, -25.63]  # tap power in dBm

    # Convert dB to linear scale for power
    powers_linear = 10 ** (np.array(powers_dB) / 10)
    num_taps = len(powers_dB)

    # Initialize the channel matrix (complex random Gaussian per subcarrier, tap, and antenna pair)
    H = np.zeros((N_sc, N_r, N_t), dtype=np.complex128)

    # Frequency range for the subcarriers
    subcarrier_frequencies = carrier_frequency + (np.arange(N_sc) - N_sc // 2) * Df   # subcarrier indices

    # Generate channel response for each tap and apply delay phase shifts
    for tap in range(num_taps):
        # Delay in seconds
        delay = delay_taps[tap] * 1e-6  # convert from microseconds to seconds

        # Apply the phase shift for each subcarrier based on the delay
        phase_shift = np.exp(-2j * np.pi * subcarrier_frequencies * delay)

        # Complex Gaussian fading for each tap, scaled by tap power
        H_tap = np.sqrt(powers_linear[tap]) * \
                (np_random.randn(N_r, N_t, N_sc) + 1j * np_random.randn(N_r, N_t, N_sc)) / np.sqrt(2)

        # Apply phase shift across subcarriers
        H += (H_tap * phase_shift).transpose(2, 0, 1)  # Adjust dimensions (N_sc, N_r, N_t)

    # Normalize channel gains
    for sc in range(N_sc):
        H[sc, :, :] /= np.linalg.norm(H[sc, :, :], ord='fro')

    return H

def plot_channel(channel, vmin=None, vmax=None, filename=None):
    global output_path

    N_sc, N_r, N_t = channel.shape

    # Only plot first receive antenna
    H = channel[:,0,:]

    dB_gain = 10 * np.log10(np.abs(H) ** 2 + 1e-5)

    # Create a normalization object
    norm = mcolors.Normalize(vmin=dB_gain.min(), vmax=dB_gain.max())

    # plt.rcParams['font.size'] = 36
    # plt.rcParams['text.usetex'] = True
    fig, ax = plt.subplots(figsize=(12, 6))

    plt.imshow(dB_gain, aspect='auto', norm=norm)

    plt.xlabel('TX Antennas')
    plt.ylabel('Subcarriers')

    plt.xticks(range(N_t))
    plt.tight_layout()

    if filename is not None:
        plt.savefig(f'{output_path}/channel_{filename}.pdf', format='pdf', dpi=fig.dpi)
        #tikzplotlib.save(f'{output_path}/channel_{filename}.tikz')
    plt.show()
    plt.close(fig)


H_a = _generate_cdl_a_channel(N_sc, N_r, N_t, frequency)
H_c = _generate_cdl_c_channel(N_sc, N_r, N_t, frequency)
H_e = _generate_cdl_e_channel(N_sc, N_r, N_t, frequency)

plot_channel(H_a)
plot_channel(H_c)
plot_channel(H_e)
