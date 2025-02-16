# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 18:20:49 2025

@author: fbm090020
"""

import numpy as np

size = 5000
tx_snr = 1  # this is a linear scale

seed = 42
random_state = np.random.RandomState(seed=seed)

qpsk_alphabet = np.array([1+1j, 1-1j, -1+1j, -1-1j])
symbol_power = np.mean(np.abs(qpsk_alphabet) ** 2)

qpsk_alphabet /= np.sqrt(symbol_power)  # power normalization step

re_qpsk = np.real(qpsk_alphabet)
im_qpsk = np.imag(qpsk_alphabet)

M = len(qpsk_alphabet)

transmit_symbol_idx = random_state.choice(range(M), size=size, replace=True)
transmit_symbols = qpsk_alphabet[transmit_symbol_idx]

noise_power = 1. / tx_snr

noise = np.sqrt(noise_power / 2.) * (random_state.normal(size=size) + \
                                     1j * random_state.normal(size=size))
    
# AWGN
receive_symbols = transmit_symbols + noise

re_transmit_symbols = np.real(transmit_symbols)
im_transmit_symbols = np.imag(transmit_symbols)
re_receive_symbols = np.real(receive_symbols)
im_receive_symbols = np.imag(receive_symbols)

import matplotlib.pyplot as plt

fig = plt.figure()
plt.scatter(re_receive_symbols, im_receive_symbols, c='r', s=4)
plt.scatter(re_qpsk, im_qpsk, c='k', s=6)
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.xlabel('I')
plt.ylabel('Q')
plt.grid(True)
plt.show()
plt.close(fig)

# Symbol detection using k-means unsupervised learning
from sklearn.cluster import KMeans
centroids = np.c_[re_qpsk, im_qpsk]
received_symbols = np.c_[re_receive_symbols, im_receive_symbols]

kmeans = KMeans(n_clusters=M, 
                init=centroids, random_state=random_state).fit(centroids)

detected_symbols_idx = kmeans.predict(received_symbols)

error = (detected_symbols_idx != transmit_symbol_idx).astype(int)
average_detection_error = np.mean(error)

print(f'k-means detection average error: {average_detection_error:.4f}')

############################
fig = plt.figure()
plt.scatter(re_receive_symbols, im_receive_symbols, cmap='jet', c=detected_symbols_idx, s=4)
plt.scatter(np.real(qpsk_alphabet), np.imag(qpsk_alphabet), c='k', s=4)
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.xlabel('I')
plt.ylabel('Q')
plt.grid(True)
plt.show()
plt.close(fig)


import pandas as pd

df = pd.DataFrame(data={'m': transmit_symbol_idx,
                        'x_I': re_transmit_symbols,
                        'x_Q': im_transmit_symbols,
                        'y_I': re_receive_symbols, 
                        'y_Q': im_receive_symbols,
                        'm_hat': detected_symbols_idx,
                        'error': error})

y = df['error'].values
X = df.drop('error', axis=1).values

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import time

clf = RandomForestClassifier(n_estimators=1, random_state=random_state)  # Try to run with 2 and 3 n_estimators

# # Define the parameter search space
# parameter_grid = {
#     'max_depth': [3, 10],
#     'n_estimators': [300, 500],    
#     'min_samples_leaf': [2, 4]
# }

# k_fold = 3
# clf = GridSearchCV(rfc, param_grid=parameter_grid, scoring='f1_weighted', cv=k_fold, n_jobs=-1)

start_time = time.time()
clf.fit(X, y)
end_time = time.time()

print(f'Grid search tuning and cross-validation time is: {(end_time - start_time):.4f} sec.')

# # Optimal based on the search of this space.
# clf_opt = clf.best_params_
# print(clf_opt)

# clf = clf.best_estimator_
# clf.fit(X, y)

y_pred = clf.predict(X)
prediction_error = (y_pred != y).astype(int)

average_classifier_prediction_error = np.mean(prediction_error)

print(f'RandomForest classifier has an error rate of: {average_classifier_prediction_error:.4f}.')
print()

importances = clf.feature_importances_  # scaled to 1.0
feature_names = df.columns[:-1]
df_feature_importance = pd.DataFrame(data={'importance': importances})
df_feature_importance['feature_name'] = df.columns[:-1].values
df_feature_importance = df_feature_importance.sort_values(by='importance', ascending=False).reset_index(drop=True)

print(df_feature_importance)
