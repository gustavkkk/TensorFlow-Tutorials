# X 와 Y 의 상관관계를 분석하는 기초적인 선형 회귀 모델을 만들고 실행해봅니다.
import tensorflow as tf

#linear regression
##Y=W*X+b
##input: pairs of (x,y)
##output: W,b
X = tf.placeholder(tf.float32,name="X")
Y = tf.placeholder(tf.float32,name="Y")
W = tf.Variable(tf.random_normal([1],-1.0,1.0))
b = tf.Variable(tf.random_normal([1],-1.0,1.0))
x_data = [1,2,3]
y_data = [2,4,6]

predict = X * W + b
cost = tf.reduce_mean(tf.square(predict - Y))
#optimizer
print("\n5.TensorFlow Optimization")
print("5.1 Linear Regression")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(cost)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 최적화를 100번 수행합니다.
    for step in range(100):
        # sess.run 을 통해 train_op 와 cost 그래프를 계산합니다.
        # 이 때, 가설 수식에 넣어야 할 실제값을 feed_dict 을 통해 전달합니다.
        _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})

        print(step, cost_val, sess.run(W), sess.run(b))

    # 최적화가 완료된 모델에 테스트 값을 넣고 결과가 잘 나오는지 확인해봅니다.
    print("\n=== Test ===")
    print("X: 5, Y:", sess.run(predict, feed_dict={X: 5}))
    print("X: 2.5, Y:", sess.run(predict, feed_dict={X: 2.5}))

