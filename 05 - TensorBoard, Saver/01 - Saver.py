# 모델을 저장하고 재사용하는 방법을 익혀봅니다.

import tensorflow as tf
import numpy as np

#load csv and split
print("\nLoad CSV and split")
data = np.loadtxt('./data.csv', delimiter=',',
                  unpack=True, dtype='float32')
x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])

#Save and Load network
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
L1 = tf.nn.relu(tf.matmul(X, W1))
W2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.))
L2 = tf.nn.relu(tf.matmul(L1, W2))
W3 = tf.Variable(tf.random_uniform([20, 3], -1., 1.))
model = tf.matmul(L2, W3)
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
print("\n5.5 Save and Load Network")
global_step = tf.Variable(0, trainable=False, name='global_step')
train_op = optimizer.minimize(cost, global_step=global_step)
model_path = './model/dnn.ckpt'
with tf.Session() as sess:
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("I don't understand why.\
              \nBut when you encounter some errors while restoring,restart the python kernel is Ok\n\
              \nIt seems like tensorflow memory managing has some problem")
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt.model_checkpoint_path)#model_path)
    else:
        sess.run(tf.global_variables_initializer())
        for step in range(2):
            sess.run(train_op, feed_dict={X: x_data, Y: y_data})
        
            print('Step: %d, ' % sess.run(global_step),
                  'Cost: %.3f' % sess.run(cost, feed_dict={X: x_data, Y: y_data}))
            
            saver.save(sess, model_path, global_step=global_step)
        #saver.save(sess, model_path)

    prediction = tf.argmax(model, 1)
    target = tf.argmax(Y, 1)
    print('predict:', sess.run(prediction, feed_dict={X: x_data}))
    print('actual:', sess.run(target, feed_dict={Y: y_data}))
    
    is_correct = tf.equal(prediction, target)
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    print('accuracy: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))
