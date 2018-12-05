import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST',one_hot = True)

#每个批次的大小
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#初始化权值
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

#初始化偏置值
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)


#卷积层
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')


#池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

with tf.name_scope('input'):
    #定义两个placeholder
    x = tf.placeholder(tf.float32,[None,784],name="x-input")
    y = tf.placeholder(tf.float32,[None,10],name="y-input")
    with tf.name_scope('x_image'):
        #改变x的格式转为4D的向量[batch,in_height,in_width,in_channels]
        x_image = tf.reshape(x,[-1,28,28,1],name="x-image")

with tf.name_scope('Conv1'):
    #初始化第一个卷积层的权值和偏置
    W_conv1 = weight_variable([5,5,1,32])
    b_conv1 = bias_variable([32])

    #把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
    h_convl = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_convl)

with tf.name_scope('Conv2'):
    #初始化第二个卷积层的权值和偏置
    W_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bias_variable([64])

    #把h_pool1和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
    h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

#通过上面的操作后得到64张7*7的平面

with tf.name_scope('fcl'):
    #初始化第一个全连接层的权值
    W_fcl = weight_variable([7*7*64,1024])#上一层有7*7*64个神经元，全连接层有1024个神经元
    b_fcl = bias_variable([1024])

    #把池化层2的输出扁平化为1维
    h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
    #求第一个全连接层的输出
    h_fcl = tf.nn.relu(tf.matmul(h_pool2_flat,W_fcl) + b_fcl)

    #keep_prob用来表示神经元的输出概率
    keep_prob = tf.placeholder(tf.float32)
    h_fcl_drop = tf.nn.dropout(h_fcl,keep_prob)

with tf.name_scope('fc2'):
    #初始化第二个全连接层
    W_fc2 = weight_variable([1024,10])#上一层有7*7*64个神经元，全连接层有1024个神经元
    b_fc2 = bias_variable([10])

    with tf.name_scope('fc2-softmax'):
        #计算输出
        prediction = tf.nn.softmax(tf.matmul(h_fcl_drop,W_fc2) + b_fc2)

with tf.name_scope('cross_entropy'):
    #交叉熵代价函数
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
    tf.summary.scalar("cross_entropy",cross_entropy)

    
with tf.name_scope('AdamOptimizer'):
    #使用AdamOptimizer进行优化
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    
with tf.name_scope('accuracy'):
    #结果存放在一个布尔列表中
    correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))#argmax返回一维张量中最大的值所在的位置
    #求准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    tf.summary.scalar("accuracy",accuracy)



# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for epoch in range(21):
#         for batch in range(n_batch):
#             batch_xs,batch_ys = mnist.train.next_batch(batch_size)
#             sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.5})
#         acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
#         print("Iter" + str(epoch) + ",Testing Accuracy = " + str(acc))
# Iter0,Testing Accuracy = 0.876
# Iter1,Testing Accuracy = 0.9672
# Iter2,Testing Accuracy = 0.977
# Iter3,Testing Accuracy = 0.9812
# Iter4,Testing Accuracy = 0.9828
# Iter5,Testing Accuracy = 0.9857
# Iter6,Testing Accuracy = 0.9862
# Iter7,Testing Accuracy = 0.9872
# Iter8,Testing Accuracy = 0.9891
# Iter9,Testing Accuracy = 0.9893
# Iter10,Testing Accuracy = 0.9889
# Iter11,Testing Accuracy = 0.9894
# Iter12,Testing Accuracy = 0.9901
# Iter13,Testing Accuracy = 0.9914
# Iter14,Testing Accuracy = 0.9907
# Iter15,Testing Accuracy = 0.9905
# Iter16,Testing Accuracy = 0.9922
# Iter17,Testing Accuracy = 0.9916
# Iter18,Testing Accuracy = 0.9898
# Iter19,Testing Accuracy = 0.9893
# Iter20,Testing Accuracy = 0.9911

merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('logs/train',sess.graph)
    test_writer = tf.summary.FileWriter('logs/test',sess.graph)
    for i in range(1001):
        #训练模型
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.5})
        #记录训练集计算的参数
        summary = sess.run(merged,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})
        train_writer.add_summary(summary,i)
        #记录测试集计算的参数
        batch_xs,batch_ys = mnist.test.next_batch(batch_size)
        summary = sess.run(merged,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})
        test_writer.add_summary(summary,i)

        if i%10==0:
            test_acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
            train_acc = sess.run(accuracy,feed_dict={x:mnist.train.images,y:mnist.train.labels,keep_prob:1.0})
            print("Iter" + str(i) + ",Testing Accuracy = " + str(test_acc)+ ",Train Accuracy = " + str(train_acc))
