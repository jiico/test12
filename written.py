import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集
mnist = input_data.read_data_sets('MNIST',one_hot=True)

#每个批次的大小，每次读取到神经网络的图片数量
batch_size = 50

#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#定义placeholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

#构建神经网络的中间层
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
Wx_plus_b_L1 = tf.matmul(x,W) + b
#激活函数tanh
L1 = tf.nn.tanh(Wx_plus_b_L1)

#定义输出层
Weights_L2 = tf.Variable(tf.random_normal([10,10]))
biases_L2 = tf.Variable(tf.zeros([10]))
Wx_plus_b_L2 = tf.matmul(L1,Weights_L2) + biases_L2
prediction = tf.nn.softmax(Wx_plus_b_L2)



#二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#使用梯度下降法（也可以使用其他优化方法）
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)


#初始化变量
init = tf.global_variables_initializer()

#预测结果存放在布尔型列表中
#argmax返回一维张量中最大值所在位置
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))

#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


with tf.Session() as sess:
    sess.run(init)
    for epoch in range(50):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})

        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})  
        print("iIter:"+str(epoch)+",Testing Accuracy :" + str(acc))  
