# 털과 날개가 있는지 없는지에 따라, 포유류인지 조류인지 분류하는 신경망 모델을 만들어봅니다.
import tensorflow as tf
import numpy as np

#Neural Network
##Input X N*2
##Hidden W,b 2*3
##Output Y N*3
## Y =
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
W = tf.Variable(tf.random_uniform([2, 3], -1., 1.))
b = tf.Variable(tf.zeros([3]))
import numpy as np
# [fur, feather]
x_data = np.array(
    [[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]])
# [else, mammal, bird]
y_data = np.array([
    [1, 0, 0],  # else
    [0, 1, 0],  # mammal
    [0, 0, 1],  # bird
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
])

L = tf.add(tf.matmul(X, W), b)
L = tf.nn.relu(L)
model = tf.nn.softmax(L)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)
print("\n5.2 One Layer Neural Network")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(100):
        sess.run(train_op, feed_dict={X: x_data, Y: y_data})
    
        if (step + 1) % 10 == 0:
            print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))
    prediction = tf.argmax(model, 1)
    target = tf.argmax(Y, 1)
    print('predict:', sess.run(prediction, feed_dict={X: x_data}))
    print('actual:', sess.run(target, feed_dict={Y: y_data}))
    
    is_correct = tf.equal(prediction, target)
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    print('accuracy: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))
