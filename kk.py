import scipy.io as io
import numpy as np
import time
from dunet_testing_v6_alis import Unet_test
noiseTime = time.time()

jjj = Unet_test()
aliasTime = time.time()
totalAliasTime = aliasTime - noiseTime
print("Totaly Aliasing time " + str(totalAliasTime) + " seconds")
#jjj = io.loadmat("/home/condauser/Documents/ao_seg/new_vel.mat")
#jjj = jjj['data']
tt = 0
nn = len(jjj)
nn = int(nn)/3
nn = int(nn)
print(jjj.shape)
print(type(jjj))
print(nn)
print(type(nn))
ddd = np.zeros([jjj[0].shape[0], jjj[0].shape[1], jjj[0].shape[2], int(3), nn])
for i in range(3):
	for j in range(int(nn)):
		ddd[:,:,:,i,j] = jjj[tt]
		tt +=1
print(ddd.shape)
