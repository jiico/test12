import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集
mnist = input_data.read_data_sets('MNIST',one_hot=True)

#每个批次的大小，每次读取到神经网络的图片数量
batch_size = 20

#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#定义placeholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
#神经元工作数量的比例
keep_prob = tf.placeholder(tf.float32)
#学习率
lr = tf.Variable(0.001,dtype=tf.float32)

#构建神经网络的中间层
W1 = tf.Variable(tf.truncated_normal([784,300],stddev=0.1))
b1 = tf.Variable(tf.zeros([300])+0.1)
L1 = tf.nn.tanh(tf.matmul(x,W1)) + b1
L1_drop = tf.nn.dropout(L1,keep_prob)


W2 = tf.Variable(tf.truncated_normal([300,100],stddev=0.1))
b2 = tf.Variable(tf.zeros([100])+0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop,W2)) + b2
L2_drop = tf.nn.dropout(L2,keep_prob)



W4 = tf.Variable(tf.truncated_normal([100,10],stddev=0.1))
b4 = tf.Variable(tf.zeros([10])+0.1)
prediction = tf.nn.softmax(tf.matmul(L2_drop,W4)+b4)


#二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#使用梯度下降法（也可以使用其他优化方法）
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
#adam
train_step = tf.train.AdamOptimizer(lr).minimize(loss)


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
        #sess.run(tf.assign(lr,0.001*0.95 ** epoch))
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.7})

        #learning_rate = sess.run(lr)
        test_acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})  
        train_acc = sess.run(accuracy,feed_dict={x:mnist.train.images,y:mnist.train.labels,keep_prob:1.0})  
        #print("Iter:"+str(epoch)+",lr :" + str(learning_rate)+",Testing Accuracy :" + str(test_acc)+",Train Accuracy :" + str(train_acc))  
        print("Iter:"+str(epoch)+",Testing Accuracy :" + str(test_acc)+",Train Accuracy :" + str(train_acc))  
