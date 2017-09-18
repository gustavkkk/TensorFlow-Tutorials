import tensorflow as tf

#variable
print("\n3.TensorFlow Variable")
W = tf.Variable(tf.random_normal([3, 2]))# initialize
b = tf.Variable(tf.random_normal([2, 1]))# initialize
sess.run(tf.global_variables_initializer())# do initialize
p2 = sess.run(W)
p3 = sess.run(b)
print(p2)
print(p3)
x = tf.placeholder(tf.float32,[None,3])# definition

#function
##y=W*x+b
##input:X,W,b
##output:y
print("\n4.Tensorflow Function")
expr = tf.matmul(x, W) + b# definition
x_data = [[1, 2, 3], [4, 5, 6]]
print("input\n",x_data)
y = sess.run(expr, feed_dict={x: x_data})#input and run
print("output\n",y)

sess.close()
