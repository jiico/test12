import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集
mnist = input_data.read_data_sets('MNIST',one_hot=True)




#每个批次的大小，每次读取到神经网络的图片数量
batch_size = 50

#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)#平均值
        with tf.name_scope('stddev'): 
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev',stddev)#标准差
        tf.summary.scalar('max',tf.reduce_max(var))#最大值
        tf.summary.scalar('min',tf.reduce_min(var))#最小值
        tf.summary.histogram('histogram',var)#直方图



#命名空间
with tf.name_scope("input"):
    #定义placeholder
    x = tf.placeholder(tf.float32,[None,784],name="x-input")
    y = tf.placeholder(tf.float32,[None,10],name="y-input")

with tf.name_scope("layer"):
    with tf.name_scope("weightssss"):
        W = tf.Variable(tf.zeros([784,10]),name='W')
        variable_summaries(W)
    with tf.name_scope("bbbbbb"):
        b = tf.Variable(tf.zeros([10]),name='b')
        variable_summaries(b)
    with tf.name_scope("Wx_plus_b_L111111111111"):
        Wx_plus_b_L1 = tf.matmul(x,W) + b

    with tf.name_scope("L1L1L1L1L1L1L1"):
        L1 = tf.nn.tanh(Wx_plus_b_L1)




#定义输出层
Weights_L2 = tf.Variable(tf.random_normal([10,10]))
biases_L2 = tf.Variable(tf.zeros([10]))
Wx_plus_b_L2 = tf.matmul(L1,Weights_L2) + biases_L2
prediction = tf.nn.softmax(Wx_plus_b_L2)



#二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
    tf.summary.scalar('loss',loss)
#使用梯度下降法（也可以使用其他优化方法）
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)


#初始化变量
init = tf.global_variables_initializer()

#预测结果存放在布尔型列表中
#argmax返回一维张量中最大值所在位置
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))

    with tf.name_scope('accuracy22'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar('accuracy',accuracy)


#合并所有的summary
merged = tf.summary.merge_all() 

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/',sess.graph)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            summary,acc = sess.run([merged,train_step],feed_dict={x:batch_xs,y:batch_ys})

        writer.add_summary(summary,epoch)

        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})  
        print("iIter:"+str(epoch)+",Testing Accuracy :" + str(acc))  
