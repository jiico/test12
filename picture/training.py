# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 13:52:06 2018

@author: admin
"""
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

train_dir = 'D:/picture/train/'

def get_files(file_dir):
    # A5 = []
    # label_A5 = []
    # Z_NG = []
    # label_A6 = []
    # SEG = []
    # label_SEG = []
    # SUM = []
    # label_SUM = []
    # LTAX1 = []
    # label_LTAX1 = []
    Z_OK = []
    label_Z_OK = []
    Z_NG = []
    label_Z_NG = []
    #定义存放各类别数据和对应标签的列表，列表名对应你所需要分类的列别名
    #A5，A6等是我的数据集中要分类图片的名字

    for file in os.listdir(file_dir):
        # name = file.split(sep='_')
        # if name[0]=='A5':
        #     A5.append(file_dir+file)
        #     label_A5.append(0)
        # elif name[0] == 'A6':
        #     A6.append(file_dir+file)
        #     label_Z_NG.append(1)
        # elif name[0]=='LTAX1':
        #     LTAX1.append(file_dir+file)
        #     label_LTAX1.append(2)
        # elif name[0] == 'SEG':
        #     SEG.append(file_dir+file)
        #     label_SEG.append(3)
        # else:
        #     SUM.append(file_dir+file)
        #     label_SUM.append(4)
        name = file.split(sep='_')
        if name[0]=='ng':
            Z_OK.append(file_dir+file)
            label_Z_OK.append(0)
        elif name[0] == 'ok':
            Z_NG.append(file_dir+file)
            label_Z_NG.append(1)
        #根据图片的名称，对图片进行提取，这里用.来进行划分
        ###这里一定要注意，如果是多分类问题的话，一定要将分类的标签从0开始。这里是五类，标签为0，1，2，3，4。我之前以为这个标签应该是随便设置的，结果就出现了Target[0] out of range的错误。

    #打印出提取图片的情况，检测是否正确提取
    print('There are %d Z_OK\nThere are %d Z_NG\n'%(len(Z_OK),len(Z_NG)))
    
    image_list = np.hstack((Z_OK,Z_NG))
    label_list = np.hstack((label_Z_OK,label_Z_NG))
    
    
    #用来水平合并数组
    temp = np.array([image_list,label_list])
    temp = temp.transpose()

    #利用shuffle打乱顺序
    np.random.shuffle(temp)
    
    #从打乱的temp中再取出list（img和lab）
    #image_list = list(temp[:, 0])
    #label_list = list(temp[:, 1])
    #label_list = [int(i) for i in label_list]
    #return image_list, label_list
    
    #将所有的img和lab转换成list
    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(i) for i in label_list]
    
    #返回两个list
    return  image_list,label_list


#step1：将上面生成的List传入get_batch() ，转换类型，产生一个输入队列queue，因为img和lab
#是分开的，所以使用tf.train.slice_input_producer()，然后用tf.read_file()从队列中读取图像
#   image_W, image_H, ：设置好固定的图像高度和宽度
#   设置batch_size：每个batch要放多少张图片
#   capacity：一个队列最大多少
def get_batch(image,label,image_W,image_H,batch_size,capacity):
    #tf.cast()用来做类型转换
    image = tf.cast(image,tf.string)
    label = tf.cast(label,tf.int32)
    
    # make an input queue加入队列
    input_queue = tf.train.slice_input_producer([image,label])
    
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])

    #step2：将图像解码，不同类型的图像不能混在一起，要么只用jpeg，要么只用png
    #jpeg或者jpg格式都用decode_jpeg函数，其他格式可以去查看官方文档 
    image = tf.image.decode_jpeg(image_contents,channels=3)
    
    
    #step3：数据预处理
    image = tf.image.resize_image_with_crop_or_pad(image,image_W,image_H)
    image = tf.reshape(image, tf.stack([image_W, image_H, 3]))

    #对resize后的图片进行标准化处理,对图像进行旋转、缩放、裁剪、归一化等操作，让计算出的模型更健壮。
    image = tf.image.per_image_standardization(image)


    #step4：生成batch
    #image_batch: 4D tensor [batch_size, width, height, 3],dtype=tf.float32 
    #label_batch: 1D tensor [batch_size], dtype=tf.int32
    image_batch,label_batch = tf.train.batch([image,label],batch_size = batch_size,num_threads=1,capacity = capacity)
    

    #重新排列label，行数为[batch_size]
    # label_batch = tf.reshape(label_batch, [batch_size])
    # image_batch = tf.cast(image_batch, tf.float32)
    label_batch = tf.reshape(label_batch,[batch_size])

    #获取两个batch，两个batch即为传入神经网络的数据
    return image_batch,label_batch
    
    


BATCH_SIZE = 10
CAPACITY = 64
IMG_W = 64
IMG_H = 64



image_list,label_list = get_files(train_dir)
image_batch,label_batch = get_batch(image_list,label_list,IMG_W,IMG_H,BATCH_SIZE,CAPACITY)

with tf.Session() as sess:
    i=0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord)
    try:
        while not coord.should_stop() and i<1:
            img,label = sess.run([image_batch,label_batch])
            
            for j in np.arange(BATCH_SIZE):
                print('label: %d'%label[j])
                #plt.imshow(img[j,:,:,:])
                # plt.show()
            i+=1
    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
    coord.join(threads)
    
    

# 建立神经网络模型，笔者的网络模型结构如下：
# 输入数据：（batch_size，IMG_W，IMG_H，col_channel）= （20,  64,  64,  3）
# 卷积层1： （conv_kernel，num_channel，num_out_neure）= （3,  3,  3,  64）
# 池化层1： （ksize，strides，padding）= （[1,3,3,1]， [1,2,2,1]， 'SAME'）
# 卷积层2： （conv_kernel，num_channel，num_out_neure）= （3,  3,  64,  16）
# 池化层2： （ksize，strides，padding）= （[1,3,3,1]， [1,1,1,1]， 'SAME'）
# 全连接1： （out_pool2_reshape，num_out_neure）= （dim， 128）
# 全连接2： （fc1_out，num_out_neure）= （128，128）
# softmax层： （fc2_out，num_classes） = （128,  4）
# 激活函数： tf.nn.relu
# 损失函数： tf.nn.sparse_softmax_cross_entropy_with_logits

#网络结构定义
    #输入参数：images，image batch、4D tensor、tf.float32、[batch_size, width, height, channels]
    #返回参数：logits, float、 [batch_size, n_classes]
def inference(images,batch_size,n_classes):
    
    #卷积层1
    #64个3x3的卷积核（3通道），padding=’SAME’，表示padding后卷积的图与原图尺寸一致，激活函数relu()
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
        
        
      
    #池化层1
    #3x3最大池化，步长strides为2，池化后执行lrn()操作，局部响应归一化，对训练有利。
    with tf.variable_scope('pooling_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1,ksize = [1,3,3,1],strides = [1,2,2,1],
                               padding = 'SAME',name = 'pooling1')
        norm1 = tf.nn.lrn(pool1,depth_radius = 4,bias = 1.0,alpha = 0.001/9.0,
                          beta = 0.75,name = 'norm1')
    
    
    #卷积层2
    #16个3x3的卷积核（16通道），padding=’SAME’，表示padding后卷积的图与原图尺寸一致，激活函数relu()
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
    
        
        
    #池化层2
    #3x3最大池化，步长strides为2，池化后执行lrn()操作，
    #pool2 and norm2
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2,depth_radius = 4,bias = 1.0,alpha = 0.001/9.0,
                          beta = 0.75,name = 'norm2')
        pool2 = tf.nn.max_pool(norm2,ksize = [1,3,3,1],strides = [1,1,1,1],
                               padding = 'SAME',name = 'pooling2')

    #全连接层3
    #128个神经元，将之前pool层的输出reshape成一行，激活函数relu()
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
    
    
    #全连接层4
    #128个神经元，激活函数relu() 
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
    
    
    #dropout层        
    #    with tf.variable_scope('dropout') as scope:
    #        drop_out = tf.nn.dropout(local4, 0.8)
                
            
    #Softmax回归层
    #将前面的FC层输出，做一个线性回归，计算出每一类的得分，在这里是2类，所以这个层输出的是两个得分。
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


#-----------------------------------------------------------------------------
#loss计算
#传入参数：logits，网络计算输出值。labels，真实值，在这里是0或者1
#返回参数：loss，损失值
def losses(logits,labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels = labels,name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy,name = 'loss')
        tf.summary.scalar(scope.name+'/loss',loss)
    return loss


#--------------------------------------------------------------------------
#loss损失值优化
#输入参数：loss。learning_rate，学习速率。
#返回参数：train_op，训练op，这个参数要输入sess.run中让模型去训练。
def trainning(loss,learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        global_step = tf.Variable(0,name = 'global_step',trainable=False)
        train_op = optimizer.minimize(loss,global_step=global_step)
    return train_op


#-----------------------------------------------------------------------
#评价/准确率计算
    #输入参数：logits，网络计算值。labels，标签，也就是真实值，在这里是0或者1。
    #返回参数：accuracy，当前step的平均准确率，也就是在这些batch中多少张图片被正确分类了。
def evaluation(logits,labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits,labels,1)
        correct = tf.cast(correct,tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+'/accuracy',accuracy)
    return accuracy



N_CLASSES = 2
IMG_W = 64
IMG_H = 64
BATCH_SIZE = 410
CAPACITY = 64
MAX_STEP = 1000
learning_rate = 0.0001

def run_training():
    train_dir = 'D:/picture/train/'
    logs_train_dir = 'D:/picture/log/'
    train,train_label = get_files(train_dir)
    train_batch,train_label_batch = get_batch(train,train_label,
                                                         IMG_W,
                                                         IMG_H,
                                                         BATCH_SIZE,
                                                         CAPACITY)
    train_logits =inference(train_batch,BATCH_SIZE,N_CLASSES)
    train_loss = losses(train_logits,train_label_batch)
    train_op = trainning(train_loss,learning_rate)
    train_acc = evaluation(train_logits,train_label_batch)
    
    
    #这个是log汇总记录
    summary_op = tf.summary.merge_all()
    
    #产生一个会话
    sess = tf.Session()

    #产生一个writer来写log文件
    train_writer = tf.summary.FileWriter(logs_train_dir,sess.graph)

    #产生一个saver来存储训练好的模型
    saver = tf.train.Saver()
    
    #所有节点初始化
    sess.run(tf.global_variables_initializer())

    #队列监控
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess,coord = coord)
    
    
    #进行batch的训练
    try:
        #执行MAX_STEP步的训练，一步一个batch
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            
            #启动以下操作节点，有个疑问，为什么train_logits在这里没有开启？
            _,tra_loss,tra_acc = sess.run([train_op,train_loss,train_acc])

            #每隔50步打印一次当前的loss以及acc，同时记录log，写入writer
            if step %  50 == 0:
                print('Step %d,train loss = %.2f,train occuracy = %.2f%%'%(step,tra_loss,tra_acc))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str,step)
            
            if step % 2000 ==0 or (step +1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir,'model.ckpt')
                saver.save(sess,checkpoint_path,global_step = step)
    except tf.errors.OutOfRangeError:
        print('Done training epoch limit reached')
    finally:
        coord.request_stop()
    
    coord.join(threads)
    sess.close()
            
run_training()   
    
    
    
    
    

# label: 4
# label: 4
# label: 4
# label: 1
# There are 5 Z_OK
# There are 6 Z_NG
# There are 3 LTAX1
# There are 5 SEG
# There are 9 SUM
# 2018-12-06 11:26:25.718034: W tensorflow/core/framework/allocator.cc:122] Allocation of 708837376 exceeds 10% of system memory.
# 2018-12-06 11:26:26.297801: W tensorflow/core/framework/allocator.cc:122] Allocation of 708837376 exceeds 10% of system memory.
# 2018-12-06 11:26:26.842604: W tensorflow/core/framework/allocator.cc:122] Allocation of 708837376 exceeds 10% of system memory.
# 2018-12-06 11:26:27.410977: W tensorflow/core/framework/allocator.cc:122] Allocation of 708837376 exceeds 10% of system memory.
# 2018-12-06 11:26:27.961331: W tensorflow/core/framework/allocator.cc:122] Allocation of 708837376 exceeds 10% of system memory.
# Step 0,train loss = 1.61,train occuracy = 1.00%
# Step 50,train loss = 3006.88,train occuracy = 0.00%
# Step 100,train loss = 430.81,train occuracy = 0.00%
# Step 150,train loss = 334.75,train occuracy = 0.00%
# Step 200,train loss = 148.38,train occuracy = 0.25%
# Step 250,train loss = 177.20,train occuracy = 0.25%
# Step 300,train loss = 211.17,train occuracy = 0.00%
# Step 350,train loss = 68.05,train occuracy = 0.25%
# Step 400,train loss = 46.72,train occuracy = 0.25%
# Step 450,train loss = 50.08,train occuracy = 0.25%
# Step 500,train loss = 35.88,train occuracy = 0.25%
# Step 550,train loss = 9.61,train occuracy = 0.00%
# Step 600,train loss = 38.04,train occuracy = 0.00%
# Step 650,train loss = 13.81,train occuracy = 0.00%
# Step 700,train loss = 2.63,train occuracy = 0.50%
# Step 750,train loss = 5.74,train occuracy = 0.25%
# Step 800,train loss = 2.96,train occuracy = 0.25%
# Step 850,train loss = 19.89,train occuracy = 0.00%
# Step 900,train loss = 2.16,train occuracy = 0.25%
# Step 950,train loss = 5.42,train occuracy = 0.50%    
    
    
