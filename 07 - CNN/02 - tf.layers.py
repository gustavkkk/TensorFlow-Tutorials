# 신경망 구성을 손쉽게 해 주는 유틸리티 모음인 tensorflow.layers 를 사용해봅니다.
# 01 - CNN.py 를 재구성한 것이니, 소스를 한 번 비교해보세요.
# 이처럼 TensorFlow 에는 간단하게 사용할 수 있는 다양한 함수와 유틸리티들이 매우 많이 마련되어 있습니다.
# 다만, 처음에는 기본적인 개념에 익숙히지는 것이 좋으므로 이후에도 가급적 기본 함수들을 이용하도록 하겠습니다.
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])
#Usage of Tensor Layer
print("\n# Tensor Layer")
is_training = tf.placeholder(tf.bool)
L1 = tf.layers.conv2d(X, 32, [3, 3])
L1 = tf.layers.max_pooling2d(L1, [2, 2], [2, 2])
L1 = tf.layers.dropout(L1, 0.7, is_training)
L2 = tf.layers.conv2d(L1, 64, [3, 3])
L2 = tf.layers.max_pooling2d(L2, [2, 2], [2, 2])
L2 = tf.layers.dropout(L2, 0.7, is_training)
L3 = tf.contrib.layers.flatten(L2)
L3 = tf.layers.dense(L3, 256, activation=tf.nn.relu)
L3 = tf.layers.dropout(L3, 0.5, is_training)
model = tf.layers.dense(L3, 10, activation=None)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    batch_size = 100
    total_batch = int(mnist.train.num_examples/batch_size)
    
    for epoch in range(15):
        total_cost = 0
    
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = batch_xs.reshape(-1, 28, 28, 1)
            _, cost_val = sess.run([optimizer, cost],
                                   feed_dict={X: batch_xs,
                                              Y: batch_ys,
                                              is_training: True})
            total_cost += cost_val
    
        print('Epoch:', '%04d' % (epoch + 1),
              'Avg. cost =', '{:.4f}'.format(total_cost / total_batch))
    
    print('optimization finished!')
    
    is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    print('accuracy:', sess.run(accuracy,
                            feed_dict={X: mnist.test.images.reshape(-1, 28, 28, 1),
                                       Y: mnist.test.labels,
                                       is_training: False}))
