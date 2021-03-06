# 텐서보드를 이용하기 위해 각종 변수들을 설정하고 저장하는 방법을 익혀봅니다.

import tensorflow as tf
import numpy as np


data = np.loadtxt('./data.csv', delimiter=',',
                  unpack=True, dtype='float32')

x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])

import shutil
import os
#Remove model directory
def removeHistory():
    print("remove model")
    if os.path.exists('./model'):
        shutil.rmtree('./model')
    if os.path.exists('./log'):   
        shutil.rmtree('./log')
removeHistory()

#Usage of Tensorboard-1.1
print("\nTensorboard")
print("\nTF scope is used for group some into a layer")
with tf.name_scope('layer1'):
    W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.), name='W1')
    L1 = tf.nn.relu(tf.matmul(X, W1))
with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.), name='W2')
    L2 = tf.nn.relu(tf.matmul(L1, W2))
with tf.name_scope('output'):
    W3 = tf.Variable(tf.random_uniform([20, 3], -1., 1.), name='W3')
    model = tf.matmul(L2, W3)
with tf.name_scope('optimizer'):
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(cost, global_step=global_step)
####
#Save variables for tensorboard
print("Save variables for tensorboard")
tf.summary.scalar('cost', cost)
####    
with tf.Session() as sess:
    saver = tf.train.Saver(tf.global_variables())   
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs', sess.graph)
    print("tensorboard --logdir=./logs\
           \nhttp://localhost:6006")
    
    for step in range(100):
        sess.run(train_op, feed_dict={X: x_data, Y: y_data})
    
        print('Step: %d, ' % sess.run(global_step),
              'Cost: %.3f' % sess.run(cost, feed_dict={X: x_data, Y: y_data}))
    
        summary = sess.run(merged, feed_dict={X: x_data, Y: y_data})
        writer.add_summary(summary, global_step=sess.run(global_step))
    
    saver.save(sess, model_path, global_step=global_step)
    
    prediction = tf.argmax(model, 1)
    target = tf.argmax(Y, 1)
    print('predict:', sess.run(prediction, feed_dict={X: x_data}))
    print('actual:', sess.run(target, feed_dict={Y: y_data}))
    
    is_correct = tf.equal(prediction, target)
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    print('accuracy: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))
