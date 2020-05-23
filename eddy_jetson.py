import h5py
import numpy as np
#from tempfile import TemporaryFile
#from collections import Counter
import os
import tensorflow as tf
#import pickle
#import random
#from pathlib import Path
#from matplotlib import pyplot as plt
import argparse
import scipy.io as io
from dense_unet_model import Unet
from lsqf2 import lsqf2
gh = []
l = []
labels = []
label = []
k = []
la = []
masks = []
images = []
f = []
parser = argparse.ArgumentParser()
parser.add_argument("--path")
args = parser.parse_args()
path = args.path
#labels = [name for name in os.listdir(".") if os.path.isdir(name)]
#print(labels)
#for name in labels:
#f = h5py.File('test.mat', 'r')
#g = h5py.File('test.mat', 'r')
#k = h5py.File('test.mat', 'r+')
flows = io.loadmat(os.path.join(os.path.sep, path, 'vel_struct.mat'))
mags = io.loadmat(os.path.join(os.path.sep, path, 'mag_struct.mat'))
flow= flows['mrStruct']
mag= mags['mrStruct']
flow = flow['dataAy']
mag = mag['dataAy']
flow = flow[0,0]
mag = mag[0,0]

imaspeed = np.power(np.mean(np.square(flow),axis=3), 0.5)
minV = np.amin(0.7*np.reshape(mag,(-1,1)))
maxV = np.amax(0.7*np.reshape(mag,(-1,1)))
tmp = mag > maxV
not_tmp = np.logical_not(tmp)
mag = np.multiply(mag, not_tmp) + np.multiply(tmp, maxV)
tmp = mag<minV
not_tmp = np.logical_not(tmp)

mag = np.multiply(mag, not_tmp) + np.multiply(tmp, minV)
mag = (mag - minV)/(maxV - minV)
pcmra = np.multiply(imaspeed, mag)
pcmra = np.squeeze(np.mean(np.square(pcmra), axis = 3))

std_flow = np.std(imaspeed, axis=3)
del flow
del mag
#print(list(f.keys()))
#print(list(g.keys()))
#tv = f['test']
#la = g['test']
#print(list(tv.keys()))
#flow = f[tv['data'][0,0]].value
#labe = g[la['truth'][0,0]].value
#mask = g[la['mask'][0,0]].value


#print(flow.dtype)
#print(labe.dtype)
#print(mask.dtype)

#print(la)
#te = k['test']
#print(list(la.keys()))
#flow = f[tv['data'][0,0]].value
#labe = g[la['data'][0,0]].value
#test = k[te['data'][0,0]].value
#print( flow.shape)

#print(tv.shape)
#for i in range(1):
#flow = f[tv['data'][i,0]].value
#labe = g[la['truth'][i,0]].value
#mask = g[la['mask'][i,0]].value
print(pcmra.shape)
print(std_flow.shape)
pcmra = np.expand_dims(pcmra,axis=3)
std_flow = np.expand_dims(std_flow, axis=3)
X = np.concatenate((pcmra, std_flow), axis=3)
X = X.astype(np.float32)
del pcmra
del std_flow

#Y = labe.transpose()
#Z = mask.transpose()
#dim1 = X.shape[0]
#dim2 = X.shape[1]
#h1 = round((dim1-160)/2)
#w2 = round((dim2-80)/2)
#X = X.astype(np.float32)
#Y = Y.astype(np.float32)
#X = X[:140,:,:]
#Y = Y[:140,:,:]
#Z = Z[:140,:,:]
for j in range(2):
	X[...,j] = (X[...,j] - np.amin(X[...,j]))/(np.amax(X[...,j]) - np.amin(X[...,j]))
#images.append(X)
#labels.append(Y)
#masks.append(Z)
    
#print(len(images))
#print(len(labels))
#print(images[0].shape)
#print(labels[0].shape)
#print(masks[0].shape)

#print(images[0].dtype)
#print(labels[0].dtype)
#print(np.unique(labels[0]))
#print(np.unique(masks[0]))



"""
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#plt.imshow(train_image[0])
#plt.show()
s = list(range(len(images)))
print(s)
#random.shuffle(s)
test_filename = 'test_eddy.tfrecords'
writer = tf.python_io.TFRecordWriter(test_filename)

for i in s:
    image_raw =  images[i].tostring()
    label_raw = labels[i].tostring()
    #mask_raw = masks[i].tostring()

    height = images[i].shape[0]
    width = images[i].shape[1]
    depth = images[i].shape[2]
    features = {'test/image': _bytes_feature(image_raw),
               'test/label': _bytes_feature(label_raw),
               #'test/mask': _bytes_feature(mask_raw),
               'test/height': _int64_feature(height),
               'test/depth':_int64_feature(depth),
               'test/width': _int64_feature(width)}
    examples = tf.train.Example(features = tf.train.Features(feature = features))
    writer.write(examples.SerializeToString())

writer.close()
"""

input_ = tf.convert_to_tensor(X, dtype=tf.float16)
input_ = tf.transpose(input_, perm=[2,0,1,3])
input_ = tf.image.resize_image_with_crop_or_pad(input_, 160, 96)
input_ = tf.transpose(input_, perm=[1,2,0,3])
#input_ = (input_ - tf.reduce_min(input_))/(tf.reduce_max(input_) - tf.reduce_min(input_))
#input_ = tf.expand_dims(input_, dim=0)
#input_ = tf.expand_dims(input_, dim=4)
input_p = tf.placeholder(tf.float32, shape=[1, 160,96,None, 2])
flag = tf.placeholder(tf.bool)
logits_1 = Unet(x = input_p, training=flag).model
saver1 = tf.train.Saver()
llogits_1 = tf.nn.softmax(logits_1)

checkpoint1 = tf.train.get_checkpoint_state("./new_eddy")
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.4
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())
	image = sess.run([input_])
	saver1.restore(sess, checkpoint1.model_checkpoint_path)
	h , seg1 = sess.run([logits_1, llogits_1], feed_dict={input_p: image, flag: True})
	#h , seg2 = sess.run([logits_1, llogits_1], feed_dict={input_1p: image[1], flag: True})
	#h , seg3 = sess.run([logits_1, llogits_1], feed_dict={input_1p: image[2], flag: True})

sess.close()
tf.keras.backend.clear_session()

mask1 = seg1[...,1]
mask1 = mask1>0.5
print(len(image))
U = image[0]
#plt.imshow(np.squeeze(U[...,16,0]), cmap=plt.get_cmap('gray'))
#plt.imshow(np.squeeze(mask1[...,16]), alpha=0.3)

#plt.show()
mask1 = np.squeeze(mask1)
print(mask1.shape)
[H, W, D] = mask1.shape
xx = np.arange(W)

xxx = np.tile(xx,[H,1])
print(xxx.shape)
yy = np.arange(H)
yy = yy[::-1]
yy = np.expand_dims(yy,axis=1)
#yy = np.fliplr([yy])[0]
#print(yy)
yyy = np.tile(yy,(1,W))
flow = flows['mrStruct']['dataAy'][0,0]
HH = flow.shape[0]
WW = flow.shape[1]
h1 = (HH - 160)/2
w1 = (WW - 96)/2
print(h1)
print(w1)
h1 = np.int(h1)
w1 = np.int(w1)
flow = flow[h1:HH-h1,w1:WW-w1,...]
#plt.imshow(np.squeeze(flow[...,16,0,5]), cmap=plt.get_cmap('gray'))
#plt.imshow(np.squeeze(mask1[...,16]), alpha=0.3)
#plt.show()

print(np.amin(flow))
print(flow.shape)
#yyy = np.fliplr(yyy)
yyy = yyy+1
xxx = xxx+1
print(yyy)
print(xxx)
new_flow = np.zeros(flow.shape)
print(new_flow.shape)
for i in range(flow.shape[2]):
	imaFlow = np.squeeze(flow[...,i,:,:])
	#print(imaFlow.shape)
	statMask = mask1[...,i]
	statMask = statMask.astype(int)
	imaFlow1 = np.zeros(imaFlow.shape)
	#statMask = statMask.astype(np.float32)
	for k in range(imaFlow.shape[2]):
		tt = flow.shape[4]
		phi, alpha, beta = lsqf2(imaFlow[...,k,tt-1],statMask,xxx,yyy)
		phi = phi.astype(np.float32)
		alpha = alpha.astype(np.float32)
		beta = beta.astype(np.float32)
		fitPlane = phi + alpha*xxx + beta*yyy
		zeroMask = imaFlow[:,:,k,tt-1] != 0
		zeroPlane = np.multiply(fitPlane, zeroMask)
		#print(zeroPlane.shape)
		zeroPlane = np.expand_dims(zeroPlane,axis=2)
		factor3D = np.tile(zeroPlane, (1,1, tt))
		imaFlow1[:,:,k,:] = np.squeeze(imaFlow[:,:,k,:]) - factor3D
		#print(phi)
		#print(alpha)
		#print(beta)
		#print(Md)
	new_flow[:,:,i,:,:] = imaFlow1
		

io.savemat('MK/new_vel.mat',{'data':new_flow})
io.savemat('MK/eddy_mask.mat',{'data':mask1})

del flow
"""
##############   Noise Masking   ##################
"""

#from dense_unet_model2 import Unet2
import tensorflow as tf

#saver = tf.train.import_meta_graph('./pre-trained/hb.ckpt-179424.meta', clear_devices=False)
graph = tf.Graph()

input_graph_def = graph.as_graph_def()
#checkpoint1 = tf.train.get_checkpoint_state("./pre-trained")


#input_ = tf.convert_to_tensor(pcmra, dtype=tf.float32)
k = np.squeeze(image[0])
print(k.shape)
#input_4 = tf.convert_to_tensor(k, dtype=tf.float32)
#input_4 = input_[32:160,...]
#input_4 = tf.image.resize_image_with_crop_or_pad(input_4,128,96)
#input_4 = (input_4 - tf.reduce_min(input_4))/(tf.reduce_max(input_4) - tf.reduce_min(input_4))
#input_4 = tf.expand_dims(input_4, dim=0)
#input_4 = tf.expand_dims(input_4, dim=4)
with graph.as_default():
	input_4 = tf.convert_to_tensor(k, dtype=tf.float16)
	input_4 = tf.expand_dims(input_4, dim=0)
	#input_4 = tf.expand_dims(input_4, dim=4)
	flag = tf.placeholder(tf.bool)

	input_2p = tf.placeholder(tf.float16, shape=[1, 160,96,None, 2])

	logits_2 = Unet(x = input_2p, training=flag).model

#logits_2 = Unet2(x = input_2p, training=flag).model
	llogits_2 = tf.nn.softmax(logits_2)
#sesss = tf.Session(graph=graph)
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.gpu_options.per_process_gpu_memory_fraction = 0.4
with tf.Session(graph=graph, config=config) as sesss:
	with graph.as_default():
		sesss.run(tf.global_variables_initializer())
		saver = tf.train.Saver(tf.global_variables())
		#checkpoint1 = tf.train.get_checkpoint_state("E:/HB/scripts_and_stuff/aliasing/new_noise")
		#sesss.run(tf.local_variables_initializer())
		s = sesss.run([input_4])
		print(s[0].shape)
		saver.restore(sesss, tf.train.get_checkpoint_state("./new_noise").model_checkpoint_path)
		h , seg3 = sesss.run([logits_2, llogits_2], feed_dict={input_2p:s[0], flag: True})
sesss.close()

mask2 = np.squeeze(seg3[...,1])
mask2 = mask2>0.5
mask2 = np.logical_not(mask2)
mask2 = mask2.astype(np.float32)
mask2 = np.expand_dims(mask2,axis=3)
mask2 = np.expand_dims(mask2,axis=4)

final_flow = np.multiply(new_flow, mask2)
mrStruct1 = flows['mrStruct']
mrStruct1['dataAy'][0,0] = final_flow

mag = mags['mrStruct']['dataAy'][0,0]
mag = mag[h1:HH-h1,w1:WW-w1,...]
mrStruct = mags['mrStruct']
mrStruct['dataAy'][0,0] = mag

print(final_flow.shape)
io.savemat('MK/vel_struct.mat',{'mrStruct':mrStruct1})
io.savemat('MK/mag_struct.mat',{'mrStruct':mrStruct})
io.savemat('MK/noise_mask.mat',{'noise':mask2})
del mrStruct1
del mrStruct
del mag
del final_flow
		#ec_off[k] = phi
		#ec_alpha[k] = alpha
		#ec_beta[k] = beta
"""
#################################
###Segmentation
#################################
"""		
import tensorflow as tf

#saver = tf.train.import_meta_graph('./pre-trained/hb.ckpt-179424.meta', clear_devices=False)
graphs = tf.Graph()

input_graph_def = graphs.as_graph_def()
#checkpoint1 = tf.train.get_checkpoint_state("./pre-trained")


#input_ = tf.convert_to_tensor(pcmra, dtype=tf.float32)
k = np.squeeze(image[0])
print(k.shape)
k = k[...,0]
#input_4 = tf.convert_to_tensor(k, dtype=tf.float32)
#input_4 = input_[32:160,...]
#input_4 = tf.image.resize_image_with_crop_or_pad(input_4,128,96)
#input_4 = (input_4 - tf.reduce_min(input_4))/(tf.reduce_max(input_4) - tf.reduce_min(input_4))
#input_4 = tf.expand_dims(input_4, dim=0)
#input_4 = tf.expand_dims(input_4, dim=4)
with graphs.as_default():
	input_5 = tf.convert_to_tensor(k, dtype=tf.float16)
	input_5 = tf.expand_dims(input_5, dim=0)
	input_5 = tf.expand_dims(input_5, dim=4)
	flag = tf.placeholder(tf.bool)

	input_3p = tf.placeholder(tf.float16, shape=[1, 160,96,None, 1])

	logits_3 = Unet(x = input_3p, training=flag).model

#logits_2 = Unet2(x = input_2p, training=flag).model
	llogits_3 = tf.nn.softmax(logits_3)
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.gpu_options.per_process_gpu_memory_fraction = 0.4
#sesss = tf.Session(graph=graph)
with tf.Session(graph=graphs, config=config) as sesss:
	with graphs.as_default():
		sesss.run(tf.global_variables_initializer())
		saver = tf.train.Saver(tf.global_variables())
		#checkpoint1 = tf.train.get_checkpoint_state("E:/HB/scripts_and_stuff/aliasing/new_noise")
		#sesss.run(tf.local_variables_initializer())
		s = sesss.run([input_5])
		print(s[0].shape)
		saver.restore(sesss, tf.train.get_checkpoint_state("./pre-trained").model_checkpoint_path)
		h , seg4 = sesss.run([logits_3, llogits_3], feed_dict={input_3p:s[0], flag: True})
sesss.close()

mask3 = np.squeeze(seg4[...,1])
mask3 = mask3>0.5

mrStruct = mags['mrStruct']
mrStruct['dataAy'][0,0] = mask3

io.savemat('MK/aorta_mask_struct.mat',{'mrStruct':mrStruct})	
del mags
del flows
del new
del mrStruct
#print(merge[0].shape)


