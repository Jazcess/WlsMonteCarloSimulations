import csv
import numpy as np
import matplotlib.pyplot as plt

# ---------------DATA-FILES-----------------

# Input data files
file_emm = open('data/Kuraray_emmission_data.csv')
file_QE = open('data/Hamamatsu_PMT_QE_data.csv')

# WLS emission spectrum data
csvreader = csv.reader(file_emm)
emms_wls = []
emms_probs = []
emms1_probs = []
count = 0

# Read and process emission spectrum data
for row in csvreader:
    if count > 1:
        emms_wls.append(float(row[0]))
        emms_probs.append(float(row[1]))
    count += 1

emms_wls = np.asarray(emms_wls)
emms_probs = np.asarray(emms_probs)
emms1_probs = emms_probs / sum(emms_probs)
emms_cdf = np.cumsum(emms1_probs)

# PMT QE data
csvreader = csv.reader(file_QE)
QE_wls = []
QE_probs = []
count = 0

# Read and process PMT Quantum Efficiency (QE) data
for row in csvreader:
    if count > 1:
        QE_wls.append(float(row[0]))
        QE_probs.append(float(row[1]))
    count += 1

QE_wls = np.asarray(QE_wls)
QE_probs = np.asarray(QE_probs)
QE_probs /= 100
