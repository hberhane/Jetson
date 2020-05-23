import scipy.io as io
import os
import numpy as np
import tensorflow as tf
from dense_unet_model import Unet
from pathlib import Path
from shutil import copyfile
import argparse
#from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#import matplotlib.pyplot as plt

#from skimage import measure
#import time
#from skimage import morphology

#from keras import backend as K 
#start_time = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("--path")
args = parser.parse_args()
path = args.path


flows = io.loadmat(os.path.join(os.path.sep, path, 'vel_struct.mat'))
mags = io.loadmat(os.path.join(os.path.sep, path, 'mag_struct.mat'))
magss = io.loadmat(os.path.join(os.path.sep, path, 'mag_struct.mat'))

try:
	flow= flows['mrStruct']
	flow = flow['dataAy']
	flow = flow[0,0]
except:
	flow = flows['dataAy']

try:
	mag= mags['mrStruct']
	mag = mag['dataAy']
	mag = mag[0,0]
except:
	mag = mags['dataAy']
	print(mag.shape)




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
h = pcmra.shape[0]
del flows
del flow
del mag
del mags
if h<160:
	input_ = tf.convert_to_tensor(pcmra, dtype=tf.float32)
	input_1 = input_[0:128,...]
	input_1 = tf.image.resize_image_with_crop_or_pad(input_1,128,96)
	h1 = (h-128)
	input_2 = input_[h1:h,...]
	input_2 = tf.image.resize_image_with_crop_or_pad(input_2,128,96)
	input_1 = (input_1 - tf.reduce_min(input_1))/(tf.reduce_max(input_1) - tf.reduce_min(input_1))
	input_1 = tf.expand_dims(input_1, dim=0)
	input_1 = tf.expand_dims(input_1, dim=4)
	input_1p = tf.placeholder(tf.float32, shape=[1, 128,96,None, 1])
	input_2 = (input_2 - tf.reduce_min(input_2))/(tf.reduce_max(input_2) - tf.reduce_min(input_2))
	input_2 = tf.expand_dims(input_2, dim=0)
	input_2 = tf.expand_dims(input_2, dim=4)
	flag = tf.placeholder(tf.bool)

	logits_1 = Unet(x = input_1p, training=flag).model
	saver1 = tf.train.Saver()

	#logits_1 = Unet(x = input_1p, training=flag).model

	llogits_1 = tf.nn.softmax(logits_1)

	checkpoint1 = tf.train.get_checkpoint_state("./pre-trained")
	
	checkpoint1 = tf.train.get_checkpoint_state("./pre-trained")
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.gpu_options.per_process_gpu_memory_fraction = 0.4


	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		image = sess.run([input_1, input_2])
		saver1.restore(sess, checkpoint1.model_checkpoint_path)
		h , seg1 = sess.run([logits_1, llogits_1], feed_dict={input_1p: image[0], flag: True})
		h , seg2 = sess.run([logits_1, llogits_1], feed_dict={input_1p: image[1], flag: True})
		#h , seg3 = sess.run([logits_1, llogits_1], feed_dict={input_1p: image[2], flag: True})

	sess.close()
	mask1 = seg1[...,1]
	mask2 = seg2[...,1]
	mask1 = mask1>0.5
	mask2 = mask2>0.5
	new = np.zeros((pcmra.shape[0], pcmra.shape[1], pcmra.shape[2]))
	mask1 = np.squeeze(mask1)
	mask2 = np.squeeze(mask2)
	w = pcmra.shape[1]
	w1 = (w-96)//2
	h = pcmra.shape[0]
	h1 = (h - 128)
	if w > 96:
		#new[0:10,w1:w-w1,:] = mask1[0:10,...]
		new[0:128,w1:(w-w1),:] = mask1
		new[128:h,w1:(w-w1),:] = mask2[128-h1:128,...]

	#new = morphology.remove_small_objects(new, min_size=500, connectivity=6)
	"""
	fig = plt.figure()
	verts, faces, normals, _ = measure.marching_cubes_lewiner(new, spacing=(2.2,2.2,2.5))
	ax = fig.ad_subplot(111,projection='3d')
	ax.axis('off')
	ax.view_init(elev=90, azim=0)
	plt.show()
	
	
	flow = np.abs(imaspeed)
	flow = np.max(np.max(flow,axis=3),axis=2)
	mask = np.max(new,axis=2)
	try:
		magsss = mag1[0,0]
	except:
		magsss = mag1['dataAt']
	magsss = np.max(np.max(magsss,axis=3),axis=2)
	mask = flow*mask
	mask = np.ma.masked_where(mask<=0, mask)
	fig2 = plt.figure()
	ax2 = fig2.add_subplot()
	im1 = ax2.imshow(magsss, cmap='grey')
	im2 = ax2.imshow(mask, cmap='jet')
	ax2.axis('off')
	fig2.colorbar(im2)
	plt.show()
	"""

	try:
		magss['mrStruct']['dataAy'][0,0] = new
		mrStruct = magss['mrStruct']
	except:
		magss['dataAy'] = new
		mrStruct = magss['dataAy']
	p = Path(path).parent
	if not os.path.exists(os.path.join(os.path.sep,p,'ML_mrStruct')):
		os.mkdir(os.path.join(os.path.sep,p,'ML_mrStruct'))
	io.savemat(os.path.join(os.path.sep, p,'ML_mrStruct','aorta_mask_struct.mat'),{'mrStruct' : mrStruct})
	del magss
	del new
	del mrStruct
	#mrStruct = flows['mrStruct']
	#io.savemat(os.path.join(os.path.sep, p,'ML_mrStruct','vel_struct.mat'),{'mrStruct' : mrStruct})
	#mrStruct = mags['mrStruct']

	#io.savemat(os.path.join(os.path.sep, p,'ML_mrStruct','mag_struct.mat'),{'mrStruct' : mrStruct})
	#copyfile(os.path.join(os.path.sep,path,'vel_struct.mat'), os.path.join(os.path.sep, p,'ML_mrStruct','vel_struct.mat'))
	#copyfile(os.path.join(os.path.sep,path,'mag_struct.mat'), os.path.join(os.path.sep, p,'ML_mrStruct','mag_struct.mat'))
elif h >= 160:


	input_ = tf.convert_to_tensor(pcmra, dtype=tf.float32)
	input_1 = input_[0:128,...]
	input_1 = tf.image.resize_image_with_crop_or_pad(input_1,128,96)
	input_2 = input_[32:160,...]
	input_2 = tf.image.resize_image_with_crop_or_pad(input_2,128,96)
	input_3 = input_[5:133,...]
	input_3 = tf.image.resize_image_with_crop_or_pad(input_3,128,96)

	input_1 = (input_1 - tf.reduce_min(input_1))/(tf.reduce_max(input_1) - tf.reduce_min(input_1))
	input_1 = tf.expand_dims(input_1, dim=0)
	input_1 = tf.expand_dims(input_1, dim=4)
	input_1p = tf.placeholder(tf.float32, shape=[1, 128,96,None, 1])
	input_2 = (input_2 - tf.reduce_min(input_2))/(tf.reduce_max(input_2) - tf.reduce_min(input_2))
	input_2 = tf.expand_dims(input_2, dim=0)
	input_2 = tf.expand_dims(input_2, dim=4)

	input_3 = (input_3 - tf.reduce_min(input_3))/(tf.reduce_max(input_3) - tf.reduce_min(input_3))
	input_3 = tf.expand_dims(input_3, dim=0)
	input_3 = tf.expand_dims(input_3, dim=4)
	flag = tf.placeholder(tf.bool)
	g = tf.Graph()
	#with tf.variable_scope("model_fn") as scope:
	logits_1 = Unet(x = input_1p, training=flag).model
	saver1 = tf.train.Saver()

	#logits_1 = Unet(x = input_1p, training=flag).model

	llogits_1 = tf.nn.softmax(logits_1)

	checkpoint1 = tf.train.get_checkpoint_state("./pre-trained")
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.gpu_options.per_process_gpu_memory_fraction = 0.4


	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		image = sess.run([input_1, input_2, input_3])
		saver1.restore(sess, checkpoint1.model_checkpoint_path)
		h , seg1 = sess.run([logits_1, llogits_1], feed_dict={input_1p: image[0], flag: True})
		h , seg2 = sess.run([logits_1, llogits_1], feed_dict={input_1p: image[1], flag: True})
		h , seg3 = sess.run([logits_1, llogits_1], feed_dict={input_1p: image[2], flag: True})

	sess.close()
	#tf.keras.backend.clear_session()
	"""
	from dense_unet_model2 import Unet2
	import tensorflow as tf

	checkpoint2= tf.train.get_checkpoint_state("./pre-trained")


	input_ = tf.convert_to_tensor(pcmra, dtype=tf.float32)

	input_2 = input_[32:160,...]
	input_2 = tf.image.resize_image_with_crop_or_pad(input_2,128,96)
	input_2 = (input_2 - tf.reduce_min(input_2))/(tf.reduce_max(input_2) - tf.reduce_min(input_2))
	input_2 = tf.expand_dims(input_2, dim=0)
	input_2 = tf.expand_dims(input_2, dim=4)
	flag = tf.placeholder(tf.bool)
	input_2p = tf.placeholder(tf.float32, shape=[1, 128,96,None, 1])
	with tf.variable_scope(scope, reuse=True):
		logits_2 = Unet2(x = input_2p, training=flag).model
		saver2 = tf.train.Saver()

	#logits_2 = Unet2(x = input_2p, training=flag).model
	llogits_2 = tf.nn.softmax(logits_2)

	with tf.Session() as sesss:
		sesss.run(tf.global_variables_initializer())
		sesss.run(tf.local_variables_initializer())
		image = sesss.run([input_2])
		saver2.restore(sesss, checkpoint2.model_checkpoint_path)
		h , seg2 = sesss.run([logits_2, llogits_2], feed_dict={input_2p:image, flag: True})
	sesss.close()
	"""
	mask1 = seg1[...,1]
	mask2 = seg2[...,1]
	mask1 = mask1>0.5
	mask2 = mask2>0.5
	mask3 = seg3[...,1]
	mask3 = mask3>0.5

	
	new = np.zeros((pcmra.shape[0], pcmra.shape[1], pcmra.shape[2]))
	mask1 = np.squeeze(mask1)
	mask2 = np.squeeze(mask2)
	mask3 = np.squeeze(mask3)
	w = pcmra.shape[1]
	w1 = (w-96)//2
	h = pcmra.shape[0]
	h1 = h - 160
	if w > 96:
		#new[0:10,w1:w-w1,:] = mask1[0:10,...]
		new[5:133,w1:(w-w1),:] = mask3
		new[133:160,w1:(w-w1),:] = mask2[101:128,...]
	else:
		new[0:10,...] = mask1[0:10,abs(w1):w-w1]
		new[5:133,w1:w-w1,:] = mask3[:,abs(w1):w-w1]

		new[133:160,...] = mask2[101:128,abs(w1):w-w1]
	#plt.imshow(mask2[...,15])
	#plt.show()
	#new = morphology.remove_small_objects(new, min_size=500, connectivity=6)
	"""
	fig = plt.figure()
	verts, faces, normals, _ = measure.marching_cubes_lewiner(new, spacing=(2.2,2.2,2.5))
	ax = fig.ad_subplot(111,projection='3d')
	ax.axis('off')
	ax.view_init(elev=90, azim=0)
	plt.show()
	
	
	flow = np.abs(imaspeed)
	flow = np.max(np.max(flow,axis=3),axis=2)
	mask = np.max(new,axis=2)
	try:
		magsss = mag1[0,0]
	except:
		magsss = mag1['dataAt']
	magsss = np.max(np.max(magsss,axis=3),axis=2)
	mask = flow*mask
	mask = np.ma.masked_where(mask<=0, mask)
	fig2 = plt.figure()
	ax2 = fig2.add_subplot()
	im1 = ax2.imshow(magsss, cmap='grey')
	im2 = ax2.imshow(mask, cmap='jet')
	ax2.axis('off')
	fig2.colorbar(im2)
	plt.show()
	"""
	try:
		magss['mrStruct']['dataAy'][0,0] = new
		mrStruct = magss['mrStruct']
	except:
		magss['dataAy'] = new
		mrStruct = magss['dataAy']
	p = Path(path).parent
	if not os.path.exists(os.path.join(os.path.sep,p,'ML_mrStruct')):
		os.mkdir(os.path.join(os.path.sep,p,'ML_mrStruct'))
	io.savemat(os.path.join(os.path.sep, p,'ML_mrStruct','aorta_mask_struct.mat'),{'mrStruct' : mrStruct})
	del magss
	del new
	del mrStruct
	#mrStruct = flows['mrStruct']
	#io.savemat(os.path.join(os.path.sep, p,'ML_mrStruct','vel_struct.mat'),{'mrStruct' : mrStruct})
	#mrStruct = mags['mrStruct']

	#io.savemat(os.path.join(os.path.sep, p,'ML_mrStruct','mag_struct.mat'),{'mrStruct' : mrStruct})
	#copyfile(os.path.join(os.path.sep,path,'vel_struct.mat'), os.path.join(os.path.sep, p,'ML_mrStruct','vel_struct.mat'))
	#copyfile(os.path.join(os.path.sep,path,'mag_struct.mat'), os.path.join(os.path.sep, p,'ML_mrStruct','mag_struct.mat'))

#print("--- %s seconds ----" % (time.time() - start_time))


