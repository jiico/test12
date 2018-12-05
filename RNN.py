import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

mnist = input_data.read_data_sets("MNIST/",one_hot=True)

#载入图片是28*28
n_inputs = 28 #输入一行，一行有28个数据
max_time = 28 #一共有28行，共28次输入
lstm_size = 100 #隐藏单元
n_classes = 10 #10个分类
batch_size = 50 #每批次50个样本
n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])


weights = tf.Variable(tf.truncated_normal([lstm_size,n_classes],stddev=0.1))
biases = tf.Variable(tf.constant(0.1,shape=[n_classes]))



#定义RNN网络
def RNN(X,weights,biases):
    #-1为50，所以变形为50,28,28
    #inputs必须为[batch_size,max_time,n_inputs]
    inputs = tf.reshape(X,[-1,max_time,n_inputs])
    #定义LSTM基本Cell
    lstm_cell = rnn.BasicLSTMCell(lstm_size)
    outputs,final_state = tf.nn.dynamic_rnn(lstm_cell,inputs,dtype=tf.float32)
    #final_state[0]表示cell state
    #final_state[1]表示hidden_state
    results = tf.nn.softmax(tf.matmul(final_state[1],weights)+biases)
    return results


#计算RNN的返回结果
prediction = RNN(x,weights,biases)
#损失函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
#使用AdamOptimizer进行优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#初始化
init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(5):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})

        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter" + str(epoch) + ",Testing Accuracy = " + str(acc))
    #保存模型
    saver.save(sess,'net/my_net.ckpt')
# Iter0,Testing Accuracy = 0.7647
# Iter1,Testing Accuracy = 0.8628
# Iter2,Testing Accuracy = 0.9009
# Iter3,Testing Accuracy = 0.9165
# Iter4,Testing Accuracy = 0.9277
# Iter5,Testing Accuracy = 0.9306
# Iter6,Testing Accuracy = 0.934
# Iter7,Testing Accuracy = 0.9385
# Iter8,Testing Accuracy = 0.9322
# Iter9,Testing Accuracy = 0.9346
# Iter10,Testing Accuracy = 0.9493
# Iter11,Testing Accuracy = 0.9511
# Iter12,Testing Accuracy = 0.9517
# Iter13,Testing Accuracy = 0.9569
# Iter14,Testing Accuracy = 0.9559
# Iter15,Testing Accuracy = 0.9587
# Iter16,Testing Accuracy = 0.9613
# Iter17,Testing Accuracy = 0.9616
# Iter18,Testing Accuracy = 0.9629
# Iter19,Testing Accuracy = 0.9635
# Iter20,Testing Accuracy = 0.9656