import tensorflow as tf
import numpy as np

#使用numpy生产100个随机点
x_data = np.random.rand(100)
y_data = x_data*0.1 + 0.2

b = tf.Variable(1111.0)
k = tf.Variable(222.0)

y = k*x_data + b

#二次代价函数
loss = tf.reduce_mean(tf.square(y_data-y))

#定义梯度下降法来进行训练的优化器
optimizer = tf.train.GradientDescentOptimizer(0.2)

#最小化代价函数
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(801):
        sess.run(train)
        if step%20 == 0 :
            print(step,sess.run([k,b]))
