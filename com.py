import tensorflow as tf
import numpy as np




#变量&常量
x = tf.Variable([1,2])
a = tf.constant([3,3])

add = tf.add(input2,input3)
sub = tf.subtract(x,a)
mul = tf.multiply(input1,add)

#运行
with tf.Session() as sess:
    result = sess.run([mul,add])

#占位符
input1 = tf.placeholder(tf.float32)

#使用numpy生产100个随机点
x_data = np.random.rand(100)

#二次代价函数
loss = tf.reduce_mean(tf.square(y_data-y))

#定义梯度下降法来进行训练的优化器
optimizer = tf.train.GradientDescentOptimizer(0.2)

#最小化代价函数
train = optimizer.minimize(loss)


#从-0.5到0.5之间生成200个点
x_data = np.linspace(-0.5,0.5,200)
#生成200行1列的矩阵
x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]