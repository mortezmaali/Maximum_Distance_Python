# -*- coding: utf-8 -*-
"""
Created on Fri May 31 20:58:42 2024

@author: Morteza
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, pinv
from sklearn.decomposition import PCA
import spectral as spy
import matplotlib.patches as mpatches

def hypermnf(cube, numComponents, mean_centered=True):
    h, w, channels = cube.shape
    cube = cube.reshape(h * w, channels)
    
    if mean_centered:
        u = np.mean(cube, axis=0)
        cube -= u

    V = np.diff(cube, axis=0)
    V = computeCov(V)
    
    U, S, _ = svd(V, full_matrices=False)
    getNonZeros = np.sum(S != 0)
    U = U[:, :getNonZeros]
    
    noiseWhitening = np.dot(U, np.diag(1 / np.sqrt(S[:getNonZeros]))).T
    
    cube = np.dot(noiseWhitening, cube.T).T
    
    pca = PCA(n_components=numComponents, svd_solver='full')
    reduced_cube = pca.fit_transform(cube)
    
    coeff = np.dot(pca.components_.T, noiseWhitening.T)
    
    return reduced_cube.reshape(h, w, numComponents), coeff

def computeCov(cube):
    return np.cov(cube, rowvar=False)

def MaxD_Endmembers(MData, MData2, N):
    Data = MData.T
    Data1 = MData2.T
    nb, pix = Data.shape
    npc, _ = Data1.shape
    magnitude = np.sum(Data**2, axis=0)
    idx1 = np.argmax(magnitude)
    idx2 = np.argmin(magnitude)
    
    Endmembers = np.zeros((npc, N))
    idxEndmembers = np.zeros(N, dtype=int)
    
    Endmembers[:, 0] = Data[:, idx1]
    Endmembers[:, 1] = Data[:, idx2]
    idxEndmembers[0] = idx1
    idxEndmembers[1] = idx2
    
    Data_proj = Data1.copy()
    Id = np.eye(npc)
    
    for k in range(2, N):
        difference1 = Data_proj[:, idx2] - Data_proj[:, idx1]
        Dps = pinv(difference1[:, np.newaxis])
        ProjD = Id - np.dot(difference1[:, np.newaxis], Dps)
        Data_proj = np.dot(ProjD, Data_proj)
        idx1 = idx2
        Res = np.sum((Data_proj[:, idx2][:, np.newaxis] - Data_proj) ** 2, axis=0)
        idx2 = np.argmax(Res)
        Endmembers[:, k] = Data[:, idx2]
        idxEndmembers[k] = idx2
        
    return Endmembers, idxEndmembers

def local_gram(Endmembers):
    return np.dot(Endmembers.T, Endmembers)

def general_gram(Endmembers):
    mean_endmembers = np.mean(Endmembers, axis=1)
    centered_endmembers = Endmembers - mean_endmembers[:, np.newaxis]
    return np.dot(centered_endmembers.T, centered_endmembers)

def cal_vol_func(Data, MNF_data, num_endmembers):
    loc_gram_fcn = np.zeros(num_endmembers)
    gen_gram_fcn = np.zeros(num_endmembers)
    
    for i in range(2, num_endmembers):
        Endmembers, _ = MaxD_Endmembers(Data, MNF_data, i + 1)
        loc_gram = local_gram(Endmembers)
        loc_gram_fcn[i] = np.sqrt(abs(np.linalg.det(loc_gram)))
        gen_gram = general_gram(Endmembers)
        gen_gram_fcn[i] = np.sqrt(abs(np.linalg.det(gen_gram)))
        
    return np.vstack((loc_gram_fcn, gen_gram_fcn))

# Main script execution
datafile = 'C:/Users/Morteza/OneDrive/Desktop/PhD/New_Data/8cal_Seurat_AFTER'
hdrfile = 'C:/Users/Morteza/OneDrive/Desktop/PhD/New_Data/8cal_Seurat_AFTER.hdr'

# Load the hyperspectral image using spectral library
hcube = spy.open_image(hdrfile)
img = hcube.load().astype(np.float64)

img = img[:, :, :151]
num_endmember = 9
I_sphere_bin = img.reshape(-1, 151)
MNFD = hypermnf(img, 151)[0].reshape(670 * 1062, 151)
fcn_array = cal_vol_func(I_sphere_bin, MNFD, num_endmember)

# Compute ratio of general gram matrix
gen_gram_matrix = fcn_array[1, :]
gen_gram_matrix = gen_gram_matrix / np.sum(gen_gram_matrix)

# Plot the volume estimation figure
endmem = np.arange(3, num_endmember + 1)
plt.figure()
plt.plot(endmem, gen_gram_matrix[2:num_endmember], linewidth=1.8)
plt.xlabel('Number of endmembers')
plt.ylabel('Estimated volume')
plt.xlim([3, num_endmember])
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

# Compute endmembers
Endmembers, endmember_index = MaxD_Endmembers(I_sphere_bin, I_sphere_bin, 9)

#Now that we have the endmembers, let's classify the image using them

m, n, l = img.shape
hcuben_reshaped = img.reshape(m * n, l)


# Estimate abundance using least squares method
abundanceMap = np.linalg.lstsq(Endmembers, hcuben_reshaped.T, rcond=-1)[0]
abundanceMap = abundanceMap.T.reshape(m, n, 9)

classNames = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7','class 8','class 9']

# Match indexes for plotting
matchIdx = np.argmax(abundanceMap, axis=2)

# Plot the abundance map
plt.figure()
for i in range(1, 10):
    plt.contourf(np.where(matchIdx == i, i, np.nan), levels=[i-0.5, i+0.5], colors=[plt.cm.tab10(i)])
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().invert_yaxis()
plt.gca().set_xticks([])
plt.gca().set_yticks([])

# Create dummy patches for legend
legend_patches = [mpatches.Patch(color=plt.cm.tab10(i), label=classNames[i-1]) for i in range(1, 10)]

# Display legend
plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()