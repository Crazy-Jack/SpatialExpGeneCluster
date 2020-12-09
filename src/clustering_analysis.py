import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram

def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))


all_genes= np.load('data.npy')
print(all_genes.shape)
flattened_array = np.reshape(all_genes,(10000,-1))
print(flattened_array.shape)
# coeffs = corr2_coeff(flattened_array,flattened_array)
# print(coeffs.shape)
Z=linkage(flattened_array, 'single', 'correlation')
print(Z)
# dendrogram(Z, color_threshold=0)

