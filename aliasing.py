from dense_unet_model import Unet
import tensorflow as tf
import numpy as np
import os
from collections import Counter
#from matplotlib import pyplot as plt
#import tensorlayer as tl
import scipy.io as io
#import statistics
#import pickle

n = 54

def feed_data():
    data_path = '/media/haben/My Passport/Autopreprocessing/WA_test_x.tfrecords'  # address to save the hdf5 file
    feature = {'test/image': tf.FixedLenFeature([], tf.string),
               #'test/label': tf.FixedLenFeature([], tf.string),
               'test/depth': tf.FixedLenFeature([], tf.int64),
               'test/height': tf.FixedLenFeature([], tf.int64),
               'test/width': tf.FixedLenFeature([], tf.int64),
               'test/venc': tf.FixedLenFeature([], tf.int64),
               'test/phases': tf.FixedLenFeature([], tf.int64)}
    
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([data_path])
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    height = tf.cast(features["test/height"], tf.int32)
    venc = tf.cast(features["test/venc"], tf.int64)
    width = tf.cast(features["test/width"], tf.int32)
    depth = tf.cast(features["test/depth"], tf.int32)
    phases = tf.cast(features["test/phases"], tf.int32)


    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['test/image'], tf.float32)
    
    # Cast label data into data type
    #label = tf.decode_raw(features['test/label'], tf.float32)
    # Reshape image data into the original shape
    image = tf.reshape(image, [height, width, depth])
    #label = tf.reshape(label, [height, width, depth])
    image = tf.image.resize_image_with_crop_or_pad(image, 160, 96)
    #label = tf.image.resize_image_with_crop_or_pad(label, 160, 96)
    #label = tf.cast(label,tf.float32)
    image = tf.cast(image,tf.float32)
    #image = image[10:138,:,:]
    #label = label[10:138,:,:]

    image = tf.expand_dims(image,axis = 0)
    #label = tf.expand_dims(label,axis = 0)
    image2 = image
    image = tf.expand_dims(image,axis = 4)
    image = (image - tf.reduce_min(image))/(tf.reduce_max(image) - tf.reduce_min(image))

    #label = tf.expand_dims(label,axis = 4)
    q = tf.FIFOQueue(capacity=1, dtypes=[tf.float32, tf.float32])
    enqueue_op = q.enqueue_many([image])
    #image, label = q.dequeue()
    qr = tf.train.QueueRunner(q,[enqueue_op])

    return image,image2, depth, venc, phases



def Unet_test():

    image_batch_placeholder = tf.placeholder(tf.float32, shape=[None, 160,96,None, 1])
    #label_batch_placeholder = tf.placeholder(tf.float32, shape=[None, 160, 96,None])
    #labels_pixels = tf.reshape(label_batch_placeholder, [-1, 1])    #   if_training_placeholder = tf.placeholder(tf.bool, shape=[])
    training_flag = tf.placeholder(tf.bool)
    image_batch,image2,depth, venc, phase = feed_data()


    #label_batch_dense = tf.arg_max(label_batch, dimension = 1)

 #   if_training = tf.Variable(False, name='if_training', trainable=False)

    logits = Unet(x = image_batch_placeholder, training=training_flag).model
    #logits = logits>1
    #logs = cost_dice(logits,label_batch_placeholder)
    llogits = tf.nn.softmax(logits)
    
    #logits = tf.argmax(logits,axis=3)

    #logits = tf.reshape(logits,(-1, 2))

    #logits_batch = tf.to_int64(tf.arg_max(logits, dimension = 3))
    #logi = tf.nn.softmax(logits)

    #N,S,W,C = logits.get_shape()

    #GH = tf.reshape(logits,[-1, 1])
    #GT = tf.nn.softmax(GT)

    #GH = tf.nn.softmax(GT, axis=1)
    #GH = tf.reshape(GT,[H.value,W.value])
    #GH = tf.argmax(GH, axis = 1)
    #GH = tf.reshape(GH,[S.value,W.value])


    #probs = tf.slice(logits,[0, 0, 0, 1], [-1, -1, -1, -1])
    #probs = tf.squeeze(probs, axis = -1)
    #y = tf.reshape(label_batch_placeholder,[-1,2])
    #y = tf.argmax(y, axis = 1)
    #y = tf.reshape(y,[S.value,W.value])
    #H = tf.cast(GH,tf.float32)
    #y = tf.cast(y,tf.float32)
    #inter = tf.reduce_sum(H * y)
    #u = tf.constant(1e-5,dtype=tf.int64)
    #union = 0.00001+ tf.reduce_sum(H) + tf.reduce_sum(y)
    #logits = tl.activation.pixel_wise_softmax(logits)
    #log = tf.reshape(logits,[-1,2])
    #log = tf.argmax(log,axis = 1)
    #log = tf.reshape(log,[S.value,W.value])



    #correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_pixels, 1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    checkpoint = tf.train.get_checkpoint_state('/media/haben/D4CC01B2CC018FC2/aliasing/new_alis1')#"/media/haben/D4CC01B2CC018FC2/alis_og_crop") #good_alis #new_alis_z #new_alis_y
    saver = tf.train.Saver()
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

 #  logits_batch = tf.to_int64(tf.arg_max(logits, dimension = 1))
    d = 0

    config = tf.ConfigProto(log_device_placement=False)
    all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])

    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tf.logging.info("Restoring full model from checkpoint file %s",checkpoint.model_checkpoint_path)
        saver.restore(sess, checkpoint.model_checkpoint_path)

        #accuracy_accu = 0

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess = sess)
        ave = []
        ma = []
        tru = []
        use = []
        gtruth = []
        low = []
        low_true = []
        low_loss = []
        new_data = []
        phases = sess.run(phase)
        print(phases)
        for i in range(int(phases)):
            image_out,alias, truth, _, Venc = sess.run([image_batch,image2, depth, venc])
            
            _, loss, llogit = sess.run([logits, logs, llogits], feed_dict={image_batch_placeholder: image_out,
                                                                                    #label_batch_placeholder: truth,
                                                                                    training_flag: True})
            
            infer_out = llogit[...,1]

            #lo = np.squeeze(soft[...,0])
            #print(logis.shape)
            data = np.squeeze(infer_out)
            #print(Venc)
            
            #print(ys.shape)
           
            #gt = np.squeeze(Y)


            mask = data.transpose()
            
            use.append(mask>0.2)
            #gt = gt.transpose()
            
           
            
            
            h = data>0.2
            #print(np.max(alias))
            alias = np.squeeze(alias)
            data = data>0.2
            data = data[...,1]
            im = np.squeeze(image_out)
            im = im[...,1]
            #plt.imshow(im)
            #plt.show()
            #plt.imshow(data)
            #plt.show()
            
           
            new_alis = alias
            """
            for i in range(alias.shape[2]):
                for x in range(alias.shape[0]):
                    for y in range(alias.shape[1]):
                        if h[x,y,i] == 1:
                            value = alias[x,y,i]
                            new = value - (np.sign(value) * Venc*2/100)
                            new_alis[x,y,i] = new

                        else:
                            continue
            new_data.append(new_alis)
            """
            mask2 = alias>=0
            mask1 = mask2 * h
            new_alis[mask1] = new_alis[mask1] - (Venc*2/100)
            mask3 = np.invert(mask2)
            mask4 = mask3 * h
            new_alis[mask4] = new_alis[mask4] + (Venc*2/100)
            new_data.append(new_alis)
            #print(np.unique(new_alis))
            #ys = (ys)
            #print(W.shape)
            
            #plt.pause(0.1)
            #plt.imshow(alias[...,2])
            #plt.show()
            #plt.imshow(mask1[...,2])
            #plt.show()
            #plt.imshow(mask4[...,2])
            #plt.show()
            #plt.pause(0.1)
            io.savemat('./new_vel.mat',{'data':new_data})

            


        

        tf.train.write_graph(sess.graph_def, 'graph/', 'my_graph.pb', as_text=False)

        coord.request_stop()
        coord.join(threads)
        sess.close()
    return new_data



def main():
    tf.reset_default_graph()

    Unet_test()



if __name__ == '__main__':
    main()
