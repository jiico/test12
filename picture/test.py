import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt



def inference(images,batch_size,n_classes):
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weight',
                                  shape = [3,3,3,16],
                                  dtype = tf.float32,
                                  initializer = tf.truncated_normal_initializer(stddev = 0.1,dtype = tf.float32))
        biases = tf.get_variable('biases',
                                 shape = [16],
                                 dtype = tf.float32,
                                 initializer = tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images,weights,strides = [1,1,1,1],padding = 'SAME')
        
        pre_activation = tf.nn.bias_add(conv,biases)
        conv1 = tf.nn.relu(pre_activation,name = scope.name)
        
        
        
    with tf.variable_scope('pooling_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1,ksize = [1,3,3,1],strides = [1,2,2,1],
                               padding = 'SAME',name = 'pooling1')
        norm1 = tf.nn.lrn(pool1,depth_radius = 4,bias = 1.0,alpha = 0.001/9.0,
                          beta = 0.75,name = 'norm1')
    
    
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weight',
                                  shape = [3,3,16,128],
                                  dtype = tf.float32,
                                  initializer = tf.truncated_normal_initializer(stddev = 0.1,dtype = tf.float32))
        biases = tf.get_variable('biases',
                                 shape = [128],
                                 dtype = tf.float32,
                                 initializer = tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1,weights,strides = [1,1,1,1],padding = 'SAME')
        pre_activation = tf.nn.bias_add(conv,biases)
        conv2 = tf.nn.relu(pre_activation,name = 'conv2')
    
        
        
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2,depth_radius = 4,bias = 1.0,alpha = 0.001/9.0,
                          beta = 0.75,name = 'norm2')
        pool2 = tf.nn.max_pool(norm2,ksize = [1,3,3,1],strides = [1,1,1,1],
                               padding = 'SAME',name = 'pooling2')

    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2,shape = [batch_size,-1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim,128],
                                  dtype=tf.float32,
                                  initializer = tf.constant_initializer(0.1))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype = tf.float32,
                                 initializer = tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape,weights)+biases,name = scope.name)
    
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[128,128],
                                  dtype=tf.float32,
                                  initializer = tf.constant_initializer(0.1))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype = tf.float32,
                                 initializer = tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        local4 = tf.nn.relu(tf.matmul(local3,weights)+biases,name = 'local4')
    
    
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[128,n_classes],
                                  dtype=tf.float32,
                                  initializer = tf.constant_initializer(0.1))
        biases = tf.get_variable('biases',
                                 shape=[n_classes],
                                 dtype = tf.float32,
                                 initializer = tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4,weights),biases,name = 'softmax_linear')
    return softmax_linear

def losses(logits,labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                        (logits=logits,labels = labels,name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy,name = 'loss')
        tf.summary.scalar(scope.name+'/loss',loss)
    return loss

def trainning(loss,learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        global_step = tf.Variable(0,name = 'global_step',trainable=False)
        train_op = optimizer.minimize(loss,global_step=global_step)
    return train_op

def evaluation(logits,labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits,labels,1)
        correct = tf.cast(correct,tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+'/accuracy',accuracy)
    return accuracy



def get_one_image(img_dir):
     image = Image.open(img_dir)
     plt.imshow(image)
     image = image.resize([64, 64])
     image_arr = np.array(image)
     return image_arr

def test(test_file):
    log_dir = 'D:/picture/log/'
    # log_dir = 'D:/AI/Project/picture/log2'
    
    image_arr = get_one_image(test_file)
    
    with tf.Graph().as_default():
        image = tf.cast(image_arr, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1,64, 64, 3])
        print(image.shape)

        plt.imshow(image_arr)
        plt.show()

        p = inference(image,1,2)

        logits = tf.nn.softmax(p)

        x = tf.placeholder(tf.float32,shape = [64,64,3])

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            ckpt = tf.train.get_checkpoint_state(log_dir)
            
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint')
            prediction = sess.run(logits, feed_dict={x: image_arr})
            max_index = np.argmax(prediction) 
            print('prediction label:')
            print(max_index)
            print('prediction result:')
            print(prediction)
            if max_index==0:
                print('This is a Z_OK with possibility %.2f' %prediction[:, 0])
            else :
                print('This is a Z_NG with possibility %.2f' %prediction[:, 1])
          

# test('D:\\picture\\test\\ng_01.jpg')
# test('D:\\picture\\test\\ng_02.jpg')
# test('D:\\picture\\test\\ng_03.jpg')
# test('D:\\picture\\test\\ng_04.jpg')
# test('D:\\picture\\test\\ng_05.jpg')
test('D:\\picture\\test\\ok_01.jpg')
# test('D:\\picture\\test\\ok_02.jpg')
# test('D:\\picture\\test\\ok_03.jpg')
# test('D:\\picture\\test\\ok_04.jpg')
# test('D:\\picture\\test\\ok_05.jpg')