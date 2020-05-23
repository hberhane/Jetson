import numpy as np
import scipy.io as io
import os

#def alias_correc(mask):
mask = io.loadmat('MK/alias_mask.mat')
mask = mask['data']
print(mask.shape)
flows = io.loadmat('MK/vel_struct.mat')
venc = flows['mrStruct']['user'][0,0]	
venc = venc['venc_in_plane'][0,0]
print(venc)
venc = venc[0][0]
print(venc)
venc = venc*100
flow = flows['mrStruct']['dataAy'][0,0]
#flow = flow[...,i]
mask2 = flow>=0
#mask2 = 1*mask2
#mask = mask*1
#mask = mask[...,i]
mask1 = mask2 * mask
flow[mask1] = flow[mask1] - (venc*2/100)
mask3 = np.invert(mask2)
#mask3 = mask3*1
#mask4 = mask3 * mask
flow[mask4] = flow[mask4] + (venc*2/100)
flows['mrStruct']['dataAy'][0,0] = flow
io.savemat('MK/vel_struct.mat',{'mrStruct':flows})

	#return venc

