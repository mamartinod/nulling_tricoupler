import numpy as np
import matplotlib.pyplot as plt

test_file = np.loadtxt("rsoft_coefficients/3DTriRatioCplLen1700Wvl14-17_Centre_bp_mon_1_last.dat")
print(test_file.shape)
test_file = np.transpose(test_file)
print(test_file.shape)
# print(test_file)

wl = test_file[0]
flux = test_file[1]
print(wl)
