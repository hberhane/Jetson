import os
import pydicom

import argparse
import glob
import numpy as np
import re
import scipy.io as io
from pathlib import Path
from shutil import copyfile
import time
start_time = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("--flow")
parser.add_argument("--mag")
args = parser.parse_args()
path = args.flow
path2 = args.mag

#print(path[0])

#parser = argparse.ArgumentParser()
#parser.add_argument("--mag")
#args2 = parser.parse_args()
#path2 = args2.path
print(path2)
def dicom_array(path, path2):
	
	flow = glob.glob(os.path.join(os.path.sep, path, '*.dcm'))
	mag = glob.glob(os.path.join(os.path.sep, path2, '*.dcm'))

	flow = sorted(flow)
	mag = sorted(mag)
	print(len(flow))
	print(len(mag))
	#mag = mag.sort()
	#print("Sorted:")


	#print(len(flow))
	#print(len(mag))
	dh = pydicom.dcmread(flow[0])
	imagePosRaw = (dh[0x00200032].value)
	#print(np.amax(dh.pixel_array))
	times = dh.CardiacNumberOfImages

	#print(time)
	numSlices = int(len(flow))/(3*times)
	#print(ds.Columns)
	#print(numSlices)
	flow_dcm = np.zeros([dh.Rows, dh.Columns, int(numSlices),3, times]);
	mag_dcm = np.zeros([dh.Rows, dh.Columns, int(numSlices), times]);
	index = 0
	for fe in range(3):
		for j in range(times):
			for k in range(int(numSlices)):
				dh = pydicom.dcmread(flow[index])
				flow_dcm[:,:,k,fe,j] = dh.pixel_array
				if fe == 0:
					dh1 = pydicom.dcmread(mag[index])
					mag_dcm[:,:,k,j] = dh1.pixel_array
				index += 1
			    #print(index)

	#print(flow_dcm.shape)
	#print(np.amax(ds.pixel_array))
	#print(ds.CardiacNumberOfImages)
	l = dh[0x00291020].value
	ll = l.decode("ascii",'ignore')
	#try:
	#    j = ll.index('sWiPMemBlock.adFree[8]')
	#except:
	#    j = ll.index('sWipMemBlock.adFree[8]')

	#print(j)
	#print(len(ll))
	#print(ll[int(len(ll))-7272:int(len(ll))-7269])
	#print(ll[86651:86654])
	#uu = (ll[86651:86654])
	#print((float(uu)*100))
	#venc = ll[j:j+60]
	#print("cats")
	#print(venc)
	venc = re.search(r"sWipMemBlock\.adFree\[8\]\s+=\s+(\d+\.\d+)",ll, re.IGNORECASE).group(1)
	#print(venc)
	venc = float(venc)
	print(venc)
	#((float(ly)*100))
	phaseRange = 4096
	tmpMask1 = np.ones(flow_dcm.shape)
	#print(tmpMask1.shape)
	tmpMask1 = tmpMask1*venc
	tmpMask2 = tmpMask1

	tmpMask1 = tmpMask1/(phaseRange/2)
	flows = flow_dcm*tmpMask1
	flows = flows - tmpMask2
	sign = [1,-1,1]
	for i in range(3):
	    flows[:,:,:,i,:] = sign[i]*flows[:,:,:,i,:]
	mag = mag_dcm
	print(mag.shape)
	#print(np.amax(mag))
	#print(np.amax(flows))
	print(flows.shape)

	try:
		os.mkdir("MK")
	except:
		print("MK")

	#Make a VALID mrstruct
	memoryType = ""
	dim1 = "size_x"
	dim2 = "size_y"
	dim3 = "size_z"
	#dim4 and #dim5 must be set based on mag/mask/vel 
	dim6 = "unused"
	dim7 = "unused"
	dim8 = "unused"
	dim9 = "unused"
	dim10 = "unused"
	dim11 = "unused"
	pixelSize = dh.PixelSpacing
	thickness = dh.SliceThickness
	#Make the vox vector
	vox = [pixelSize[1], pixelSize[0], thickness, 0]
	orient = ""
	method = ""
	te = dh.EchoTime
	tr = dh.RepetitionTime
	ti = []
	patient = str(dh[0x00100010].value)
	user_size_t_step = tr
	user_encoding_type = "velocity"
	user_venc_in_plane = venc
	user_venc_through_plane = venc 
	user_nominal_interval = int(dh.NominalInterval)
	#Calculate the edges matrix
	imageOrientRaw = (dh[0x00200037].value)
	#ImagePositonRaw set earlier on first image
	#Construct edges
	imageOrient = [float(ii) for ii in imageOrientRaw]
	imagePos = [float(ii) for ii in imagePosRaw]
	del imagePosRaw, imageOrientRaw

	#Make the edges array (from dicom_read_singlefile.m
	#imageOrient = np.reshape(imageOrient,(3,2))
	imageOrient = np.array([[imageOrient[0],imageOrient[3]],[imageOrient[1],imageOrient[4]],[imageOrient[2],imageOrient[5]]])
	edges = np.diag(np.ones(4))
	edges[0:3,0]=imageOrient[:,0]*vox[0]
	edges[0:3,1]=imageOrient[:,1]*vox[1]
	# Multiplication by -1 not sure why but this makes it equal edges
	edges[0:3,2]=-1*np.cross(edges[0:3,0]/np.linalg.norm(edges[0:3,0]), edges[0:3,1]/np.linalg.norm(edges[0:3,1])*vox[2])
	edges[0:3,3]=imagePos;
	edges[:,[0,1]] = edges[:,[1,0]]

	#Save the mrStructs
	io.savemat("MK/vel_struct.mat",{"mrStruct":{"dataAy":flows,"memoryType":memoryType,"dim1":dim1,"dim2":dim2,"dim3":dim3,"dim4":"echoes","dim5":"size_t","dim6":dim6,"dim7":dim7,"dim8":dim8,"dim9":dim9,"dim10":dim10,"dim11":dim11,"vox":vox,"edges":edges,"orient":orient,"method":method,"te":te,"tr":tr,"ti":ti,"patient":patient,"user":{"size_t_step":tr,"encoding_type":user_encoding_type,"venc_in_plane":user_venc_in_plane,"venc_through_plane":user_venc_through_plane,"nominal_interval":user_nominal_interval}}})
	io.savemat("MK/mag_struct.mat",{"mrStruct":{"dataAy":mag,"memoryType":memoryType,"dim1":dim1,"dim2":dim2,"dim3":dim3,"dim4":"size_t","dim5":"unused","dim6":dim6,"dim7":dim7,"dim8":dim8,"dim9":dim9,"dim10":dim10,"dim11":dim11,"vox":vox,"edges":edges,"orient":orient,"method":method,"te":te,"tr":tr,"ti":ti,"patient":patient,"user":{"size_t_step":tr,"encoding_type":user_encoding_type,"venc_in_plane":user_venc_in_plane,"venc_through_plane":user_venc_through_plane,"nominal_interval":user_nominal_interval}}})
	return path

#del flows
#del mag
#del flow
#del mag_dcm
#del flow_dcm
print(locals())
path = dicom_array(path, path2)
os.system("python /home/condauser/Documents/ao_seg/eddy_jetson.py --path=/home/condauser/Documents/ao_seg/MK")

#os.system("python /home/condauser/Documents/ao_seg/aorta_seg.py --path=/home/condauser/Documents/ao_seg/MK")
j = Path(path).parent
try:
	os.mkdir(os.path.join(os.path.sep,j,"3dpcML"))
except:
	print(os.path.join(os.path.sep,j,"3dpcML"))
copyfile("/home/condauser/Documents/ao_seg/MK/vel_struct.mat",os.path.join(os.path.sep,j,"3dpcML","vel_struct.mat"))
copyfile("/home/condauser/Documents/ao_seg/MK/mag_struct.mat",os.path.join(os.path.sep,j,"3dpcML","mag_struct.mat"))
copyfile("/home/condauser/Documents/ao_seg/MK/aorta_mask_struct.mat",os.path.join(os.path.sep,j,"3dpcML","aorta_mask_struct.mat"))
print("--- %s seconds ----" % (time.time() - start_time))

