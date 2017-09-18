import tensorflow as tf

# constant
print("1.TensorFlow Constant")
string = tf.constant("I started learning tensorflow.")
a = tf.constant(12)
b = tf.constant(26)
c = tf.add(a,b)

#tf value 2 python value
print("\n2.TF2Python")
print(string)
sess = tf.Session()
print(sess.run(string))
d = sess.run(c)
print(d)

sess.close()
