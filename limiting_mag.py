import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
import matplotlib.mlab as mlab

FILE_NAME = 'FLUX_data_50-100.txt'
FLUX = np.loadtxt(FILE_NAME ,usecols=(0))

x_max = 100
x_min = -50

counts_ceiling = 70
norm_counts_ceiling = 0.04

##############################   Figure 1    ################################ 
plt.subplot(2,1,1)

plt.hist(FLUX, bins=range(-200, 200))

plt.axis([x_min, x_max, 0, counts_ceiling])
#plt.xlabel('Flux (counts)')
plt.ylabel('Frequency')
plt.title('Histogram')
##############################   Figure 2    ################################ 
plt.subplot(2,1,2)

n, bins, patches = plt.hist(FLUX, bins=range(-200, 200), normed=1, alpha=0.75)

plt.axis([x_min, x_max, 0, norm_counts_ceiling])
plt.xlabel('Flux (counts)')
plt.ylabel('Frequency')

plt.tight_layout
plt.show()
##############################   Figure 3    ################################ 
plt.subplot(3,1,1)

n, bins, patches = plt.hist(FLUX, bins=range(-50, 100), normed=1)
(mean, sigma) = norm.fit(FLUX)
y = mlab.normpdf(bins, mean, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)

plt.axis([x_min, x_max, 0, norm_counts_ceiling])
#plt.xlabel('Flux (counts)')
plt.ylabel('Frequency')
plt.title('Gaussian distribution fitting')
##############################   Figure 4    ################################ 
plt.subplot(3,1,2)

FLUX_new = []
for k in range(len(FLUX)):
    if FLUX[k] < mean + 3*sigma:
        FLUX_new.append(FLUX[k])

(mean_new, sigma_new) = norm.fit(FLUX_new)
n, bins, patches = plt.hist(FLUX_new, bins=range(-50, 100), normed=1)

y_new = mlab.normpdf(bins, mean_new, sigma_new)
l_new = plt.plot(bins, y, 'r--', linewidth=2)

plt.axis([-50, 100, 0, norm_counts_ceiling])
#plt.xlabel('Flux (counts)')
plt.ylabel('Frequency')
##############################   Figure 5    ################################ 
plt.subplot(3,1,3)

FLUX_new_2 = []
for i in range(len(FLUX_new)):
    if FLUX_new[i] < mean_new + 3*sigma_new:
        FLUX_new_2.append(FLUX_new[i])

(mean_new_2, sigma_new_2) = norm.fit(FLUX_new_2)
n, bins, patches = plt.hist(FLUX_new_2, bins=range(-50, 100), normed=1)

y_new_2 = mlab.normpdf(bins, mean_new_2, sigma_new_2)
l_new_2 = plt.plot(bins, y, 'r--', linewidth=2)

plt.axis([-50, 100, 0, norm_counts_ceiling])
plt.xlabel('Flux (counts)')
plt.ylabel('Frequency')

plt.tight_layout
plt.show()
##############################   Figure 6    ################################ 
l = plt.plot(bins, y, 'r', linewidth=2)
l_new = plt.plot(bins, y_new, 'g', linewidth=2)
l_new_2 = plt.plot(bins, y_new_2, 'b', linewidth=2)
n, bins, patches = plt.hist(FLUX, bins=range(-50, 100), normed=1, alpha=0.3)

plt.xlabel('Flux (counts)')
plt.ylabel('Frequency')

plt.title('Gaussian distribution fitting')

plt.show()
##############################################################################
print(mean, sigma)
print(mean_new, sigma_new)
print(mean_new_2, sigma_new_2)
print(abs(mean_new-mean)/mean)
print(abs(mean_new_2-mean_new)/mean_new)   

limiting_magnitude = 32 - 2.5*np.log10(5*sigma_new_2)

print('Limiting magnitude = %.3f'%limiting_magnitude)