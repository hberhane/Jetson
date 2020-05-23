import h5py
import numpy as np
from tempfile import TemporaryFile
from collections import Counter
import os
import tensorflow as tf
#import pickle
from pathlib import Path


#pa = Path("/media/haben/My Passport/Autopreprocessing/")
def TFrecord(mrStruct, user_venc_in_plane):
	images = []

	Venc = []
	data = mrStruct['dataAy'][0,0]

	#venc = mrStruct['user'][0]['venc_in_plane'][0]
	venc = mrStruct['user'][0,0]
	venc = venc['venc_in_plane'][0,0]
	print(venc)
	venc = venc[0][0]
	print(venc)
	venc = venc*100

	#print(tv.shape)
	for i in range(np.int(data.shape[3])):
		for j in range(np.int(data.shape[4])):
		
			flow = np.squeeze(data[:,:,:,i,j])
			
			#X = flow.transpose()
			#Y = flow.transpose()
			#dim1 = X.shape[0]
			#dim2 = X.shape[1]
			#h1 = round((dim1-160)/2)
			#w2 = round((dim2-80)/2)
			#X = X[:140,:,:]
			#Y = Y[:140,:,:]
			X = flow
			X = X.astype(np.float32)
			#venc = user_venc_in_plane
			venc = venc.astype(np.int32)
			#Y = Y.astype(np.float32)
			images.append(X)
			#labels.append(Y)
			Venc.append(venc)

	print(len(images))
	#print(len(labels))
	print(images[0].shape)
	#print(labels[0].shape)
	print(images[0].dtype)
	#print(labels[0].dtype)
	print(Venc[0])
	t = Venc[0]
	#print(np.unique(labels[0]))



	def _bytes_feature(value):
	  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

	def _int64_feature(value):
	  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

	#plt.imshow(train_image[0])
	#plt.show()
	s = list(range(len(images)))
	#random.shuffle(s)
	test_filename = 'WA_test_x.tfrecords'
	writer = tf.python_io.TFRecordWriter(test_filename)

	for i in s:
		image_raw =  images[i].tostring()
		#label_raw = labels[i].tostring()
		height = images[i].shape[0]
		width = images[i].shape[1]
		depth = images[i].shape[2]
		venc = Venc[i]
		Phases = len(images)
		features = {'test/image': _bytes_feature(image_raw),
				   #'test/label': _bytes_feature(label_raw),
				   'test/height': _int64_feature(height),
				   'test/depth':_int64_feature(depth),
				   'test/width': _int64_feature(width),               
				   'test/venc': _int64_feature(venc),
				   'test/phases': _int64_feature(Phases)}
		examples = tf.train.Example(features = tf.train.Features(feature = features))
		writer.write(examples.SerializeToString())

	writer.close()
	return t


#print(merge[0].shape)


