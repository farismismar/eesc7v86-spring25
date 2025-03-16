#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 20:54:45 2025

@author: farismismar
"""

import numpy as np 

np_random = np.random.RandomState(seed=42)

N_sc = 64
N_r = 2
N_t = 2 
sigma_dB = 4
frequency = 1800e6


def _generate_cdl_a_channel(N_sc, N_r, N_t, sigma_dB, carrier_frequency, delay_spread=5e-6, num_clusters=23, cASD=5.0, cASA=11.0, cZSD=3.0, cZSA=3.0, xpr=10.0):
    global np_random

    # Channel parameters from CDL-A model (delays, powers, AoD, AoA)
    delay_taps = np.array([0.0, 0.3819, 0.4025, 0.5868, 0.4610, 0.5375, 0.6708, 0.5750, 0.7618, 1.5375, 1.8978, 2.2242, 2.1718, 2.4942, 2.5119, 3.0582, 4.0810, 4.4579, 4.5695, 4.7966, 5.0066, 5.3043, 9.6586])
    powers_dB = np.array([-13.4, 0.0, -2.2, -4.0, -6.0, -8.2, -9.9, -10.5, -7.5, -15.9, -6.6, -16.7, -12.4, -15.2, -10.8, -11.3, -12.7, -16.2, -18.3, -18.9, -16.6, -19.9, -29.7])
    aod = np.array([-178.1, -4.2, -4.2, -4.2, 90.2, 90.2, 90.2, 121.5, -81.7, 158.4, -83.0, 134.8, -153.0, -172.0, -129.9, -136.0, 165.4, 148.4, 132.7, -118.6, -154.1, 126.5, -56.2])
    aoa = np.array([51.3, -152.7, -152.7, -152.7, 76.6, 76.6, 76.6, -1.2, 10.5, -45.6, 88.1, 34.5, 45.0, -28.4, -90.2, 12.8, -9.7, 37.4, 28.2, 15.7, 3.0, 5.0, 16.0])

    num_taps = len(powers_dB)
    
    # Initialize MIMO OFDM channel matrix H with dimensions (N_sc, N_r, N_t)
    H = np.zeros((N_sc, N_r, N_t), dtype=np.complex128)

    # Apply shadow fading (log-normal) to the large-scale fading
    shadow_fading = 10 ** (np_random.normal(0, sigma_dB, size=(N_sc, N_r, N_t)) / 10)

    # Frequency range for the subcarriers
    subcarrier_frequencies = np.arange(N_sc) / N_sc  # normalized subcarrier indices
    
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
            # Complex Gaussian fading for each tap, scaled by tap power, and shadow fading

            H_tap = np.sqrt(power * shadow_fading[sc, :, :]) * \
                                 np.outer(np.exp(1j * aod_rad), np.exp(1j * aoa_rad)) * \
                                     (np_random.randn(N_sc, N_r, N_t) + 1j * np_random.randn(N_sc, N_r, N_t)) / np.sqrt(2)
    
            # Apply phase shift across subcarriers
            H += H_tap * phase_shift[sc]
        
    return H


def _generate_cdl_c_channel_v2(N_sc, N_r, N_t, sigma_dB, carrier_frequency):
    global np_random

    # Generates a 3GPP 38.900 CDL-C channel with dimensions (N_sc, N_r, N_t).
    delay_taps = [0, 0.209, 0.423, 0.658, 1.18, 1.44, 1.71]  # in microseconds
    powers_dB = [-0.2, -13.5, -15.4, -18.1, -20.0, -22.1, -25.2]  # tap power in dBm

    num_taps = len(powers_dB)

    # Initialize the channel matrix (complex random Gaussian per subcarrier, tap, and antenna pair)
    H = np.zeros((N_sc, N_r, N_t), dtype=np.complex128)

    # Apply shadow fading (log-normal) to the large-scale fading
    shadow_fading = 10 ** (np_random.normal(0, sigma_dB, size=(N_sc, N_r, N_t)) / 10)

    # Frequency range for the subcarriers
    subcarrier_frequencies = np.arange(N_sc) / N_sc  # normalized subcarrier indices

    # Generate channel response for each tap and apply delay phase shifts
    for tap in range(num_taps):
        # Delay in seconds
        delay = delay_taps[tap] * 1e-6  # convert from microseconds to seconds
        power = 10 ** (powers_dB[tap] / 10.)  # Linear scale of power
    
        # Apply the phase shift for each subcarrier based on the delay
        phase_shift = np.exp(-2j * np.pi * subcarrier_frequencies * delay)
        
        # For each subcarrier and symbol, calculate the MIMO channel response
        for sc in range(N_sc):
            # Generate the channel matrix for this subcarrier and symbol
            # For each antenna, the channel response is influenced by the AoD and AoA
            # Complex Gaussian fading for each tap, scaled by tap power, and shadow fading
            H_tap = np.sqrt(power * shadow_fading[sc, :, :]) * \
                    (np_random.randn(N_sc, N_r, N_t) + 1j * np_random.randn(N_sc, N_r, N_t)) / np.sqrt(2)
    
            # Apply phase shift across subcarriers
            H += H_tap * phase_shift[sc]
            
    return H



def _generate_cdl_e_channel_v2(N_sc, N_r, N_t, sigma_dB, carrier_frequency, delay_spread=5e-6, num_clusters=23, cASD=5.0, cASA=11.0, cZSD=3.0, cZSA=3.0, xpr=10.0):
    global np_random

    # Channel parameters from CDL-A model (delays, powers
    delay_taps = [0, 0.264, 0.366, 0.714, 1.53, 1.91, 3.52, 4.20, 5.35]  # in microseconds
    powers_dB = [-0.03, -4.93, -8.03, -10.77, -15.86, -18.63, -21.11, -22.50, -25.63]  # tap power in dBm
    
    num_taps = len(powers_dB)
    
    # Initialize MIMO OFDM channel matrix H with dimensions (N_sc, N_r, N_t)
    H = np.zeros((N_sc, N_r, N_t), dtype=np.complex128)

    # Apply shadow fading (log-normal) to the large-scale fading
    shadow_fading = 10 ** (np_random.normal(0, sigma_dB, size=(N_sc, N_r, N_t)) / 10)

    # Frequency range for the subcarriers
    subcarrier_frequencies = np.arange(N_sc) / N_sc  # normalized subcarrier indices
    
    # Generate channel response for each tap and apply delay phase shifts
    for tap in range(num_taps):
        # Delay in seconds
        delay = delay_taps[tap] * 1e-6  # convert from microseconds to seconds
        power = 10 ** (powers_dB[tap] / 10.)  # Linear scale of power
    
        # Apply the phase shift for each subcarrier based on the delay
        phase_shift = np.exp(-2j * np.pi * subcarrier_frequencies * delay)
        
        # For each subcarrier and symbol, calculate the MIMO channel response
        for sc in range(N_sc):
            # Generate the channel matrix for this subcarrier and symbol
            # For each antenna, the channel response is influenced by the AoD and AoA
            # Complex Gaussian fading for each tap, scaled by tap power, and shadow fading
            H_tap = np.sqrt(power * shadow_fading[sc, :, :]) * \
                    (np_random.randn(N_sc, N_r, N_t) + 1j * np_random.randn(N_sc, N_r, N_t)) / np.sqrt(2)
    
            # Apply phase shift across subcarriers
            H += H_tap * phase_shift[sc]
        
    return H



H_a = _generate_cdl_a_channel(N_sc, N_r, N_t, sigma_dB, frequency)
H_c = _generate_cdl_c_channel_v2(N_sc, N_r, N_t, sigma_dB, frequency)
H_e = _generate_cdl_e_channel_v2(N_sc, N_r, N_t, sigma_dB, frequency)

