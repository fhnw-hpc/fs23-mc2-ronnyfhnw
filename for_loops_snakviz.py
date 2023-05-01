import numpy as np
import os
import imageio
from datetime import datetime

# load images

subfolder = '001'
folders = os.path.join('adni_png', subfolder)

images = np.empty([4,256,170])
idx = 0
names = []
for filename in os.listdir(folders):
    if filename.endswith('.png') and '145' in filename and '1_slice' in filename:
        with open(os.path.join(folders, filename), 'r') as f:
            im = imageio.imread(f.name)
            names.insert(idx,f.name[-17:-4])
            images[idx,:,:] = im
            print (names[idx], im.shape)
            idx += 1

def reconstruct_svd_for_loops2(u,s,vt,k):
    """SVD reconstruction for k components using 2 for-loops
    
    Inputs:
    u: (m,n) numpy array
    s: (n) numpy array (diagonal matrix)
    vt: (n,n) numpy array
    k: number of reconstructed singular components
    
    Ouput:
    (m,n) numpy array U_mk * S_k * V^T_nk for k reconstructed components
    """
    ### BEGIN SOLUTION
    reco = np.zeros((u.shape[0], u.shape[1]))
    
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            reco[i, j] = u[i, 0:k] * s[0:k] @ vt[0:k, j]
    ### END SOLUTION

    return reco

times = np.array([])
for _ in range(10):
    u, s, vt = np.linalg.svd(images[0], full_matrices=False)
    start = datetime.now()
    reconstruct_svd_for_loops2(u,s,vt,170)
    times = np.append(times, (datetime.now() - start).total_seconds())

print(times.mean())