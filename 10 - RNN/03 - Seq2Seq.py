# 챗봇, 번역, 이미지 캡셔닝등에 사용되는 시퀀스 학습/생성 모델인 Seq2Seq 을 구현해봅니다.
# 영어 단어를 한국어 단어로 번역하는 프로그램을 만들어봅니다.
import tensorflow as tf
import numpy as np

# S: 디코딩 입력의 시작을 나타내는 심볼
# E: 디코딩 출력을 끝을 나타내는 심볼
# P: 현재 배치 데이터의 time step 크기보다 작은 경우 빈 시퀀스를 채우는 심볼
#    예) 현재 배치 데이터의 최대 크기가 4 인 경우
#       word -> ['w', 'o', 'r', 'd']
#       to   -> ['t', 'o', 'P', 'P']
#RNN-3
char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz단어나무놀이소녀키스사랑']
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)

seq_data = [['word', '단어'], ['wood', '나무'],
            ['game', '놀이'], ['girl', '소녀'],
            ['kiss', '키스'], ['love', '사랑']]

def make_batch(seq_data):
    input_batch = []
    output_batch = []
    target_batch = []

    for seq in seq_data:
        input = [num_dic[n] for n in seq[0]]
        output = [num_dic[n] for n in ('S' + seq[1])]
        target = [num_dic[n] for n in (seq[1] + 'E')]

        input_batch.append(np.eye(dic_len)[input])
        output_batch.append(np.eye(dic_len)[output])
        target_batch.append(target)

    return input_batch, output_batch, target_batch

learning_rate = 0.01
n_hidden = 128
total_epoch = 100
n_class = n_input = dic_len

print("\
\n#EL: Encode layer\
\n#DL: Decode layer\
\n#T: Transpose\
\n#Input:dic_len\
\n#Step:Unknown\
\n#EL:(Unknown*dic_len)*128\
\n#Input:dic_len\
\n#Step:Unknown\
\n#DL:(Unknown*dic_len)*128\
\n#T:?\
\n#L2:128*26\
\n#Output:26\
#\
")

enc_input = tf.placeholder(tf.float32, [None, None, n_input])
dec_input = tf.placeholder(tf.float32, [None, None, n_input])
targets = tf.placeholder(tf.int64, [None, None])

with tf.variable_scope('encode'):
    enc_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.5)
    outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_input,
                                            dtype=tf.float32)

with tf.variable_scope('decode'):
    dec_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.5)
    outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_input,
                                            initial_state=enc_states,#?????
                                            dtype=tf.float32)

model = tf.layers.dense(outputs, n_class, activation=None)
cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=model, labels=targets))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) 
    input_batch, output_batch, target_batch = make_batch(seq_data)
    
    for epoch in range(total_epoch):
        _, loss = sess.run([optimizer, cost],
                           feed_dict={enc_input: input_batch,
                                      dec_input: output_batch,
                                      targets: target_batch})
    
        print('Epoch:', '%04d' % (epoch + 1),
              'cost =', '{:.6f}'.format(loss))
    
    print('optimization finished!')
      
    def translate(word):
        seq_data = [word, 'P' * len(word)]    
        input_batch, output_batch, target_batch = make_batch([seq_data])
    
        prediction = tf.argmax(model, 2)
    
        result = sess.run(prediction,
                          feed_dict={enc_input: input_batch,
                                     dec_input: output_batch,
                                     targets: target_batch})
    
        decoded = [char_arr[i] for i in result[0]]    
        end = decoded.index('E')
        translated = ''.join(decoded[:end])
    
        return translated
    
    
    print('\n=== translation test ===')
    
    print('word ->', translate('word'))
    print('wodr ->', translate('wodr'))
    print('love ->', translate('love'))
    print('loev ->', translate('loev'))
    print('abcd ->', translate('abcd'))
