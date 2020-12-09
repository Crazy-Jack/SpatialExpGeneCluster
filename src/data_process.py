'''To convert the data into 10,000 x 200 x 200 x 7 shape where there are 10,000 genes ,
200 is the max X and Y coordinate and 7 plates'''
'''OB_expression has 10,000 rows'''

import numpy as np

coords = np.loadtxt('seqfish/cell_locations/OB_centroids_coord.txt',skiprows=1, usecols=(1,2))
# converting x and y to 0-200 range
coords[:,0] = np.subtract(coords[:,0],coords[:,0].min()) * 31/(coords[:,0].max() - coords[:,0].min())
coords[:,1] = np.subtract(coords[:,1],coords[:,1].min()) * 31/(coords[:,1].max() - coords[:,1].min())
print(coords.shape)

channel_info = np.loadtxt('seqfish/cell_locations/OB_centroids_annot.txt',skiprows=1, usecols=(1),dtype=int)
print(channel_info.shape)
print(channel_info.max())

exp_matrix = np.loadtxt('seqfish/count_matrix/OB_expression.txt', usecols=range(1,2051),skiprows=1)
print(exp_matrix.shape)

final_matrix = np.zeros((exp_matrix.shape[0],32,32,7))
for row in range(exp_matrix.shape[0]):
    final_matrix[row, coords[:,0].astype(int),coords[:,1].astype(int),channel_info] += exp_matrix[row,:]

print(final_matrix.shape)
np.save('data.npy',final_matrix)