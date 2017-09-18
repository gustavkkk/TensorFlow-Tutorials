# Word2Vec 모델을 간단하게 구현해봅니다.
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#Word2Vec
import matplotlib
import matplotlib.pyplot as plt
font_name = matplotlib.font_manager.FontProperties(
                #fname="/Library/Fonts/NanumGothic.otf"
                fname='D:/Util/Font/Pwpubume.ttf'
            ).get_name()
matplotlib.rc('font', family=font_name)

sentences = ["나 고양이 좋다",
             "나 강아지 좋다",
             "나 동물 좋다",
             "강아지 고양이 동물",
             "여자친구 고양이 강아지 좋다",
             "고양이 생선 우유 좋다",
             "강아지 생선 싫다 우유 좋다",
             "강아지 고양이 눈 좋다",
             "나 여자친구 좋다",
             "여자친구 나 싫다",
             "여자친구 나 영화 책 음악 좋다",
             "나 게임 만화 애니 좋다",
             "고양이 강아지 싫다",
             "강아지 고양이 좋다"]
word_list = " ".join(sentences).split()#['나', '고양이', '좋다', '나', '강아지', '좋다', '나', '동물', '좋다', '강아지', '고양이', '동물', '여자친구', '고양이', '강아지', '좋다', '고양이', '생선', '우유', '좋다', '강아지', '생선', '싫다', '우유', '좋다', '강아지', '고양이', '눈', '좋다', '나', '여자친구', '좋다', '여자친구', '나', '싫다', '여자친구', '나', '영화', '책', '음악', '좋다', '나', '게임', '만화', '애니', '좋다', '고양이', '강아지', '싫다', '강아지', '고양이', '좋다']
word_list = list(set(word_list))#['싫다', '여자친구', '나', '눈', '영화', '강아지', '좋다', '우유', '고양이', '만화', '생선', '게임', '애니', '음악', '동물', '책']
word_dict = {w: i for i, w in enumerate(word_list)}#{'싫다': 0, '여자친구': 1, '나': 2, '영화': 4, '강아지': 5, '눈': 3, '좋다': 6, '고양이': 8, '생선': 10, '게임': 11, '만화': 9, '애니': 12, '음악': 13, '동물': 14, '책': 15, '우유': 7}
word_index = [word_dict[word] for word in word_list]#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
####
skip_grams = []#[[1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3], [4, 5], [5, 4], [5, 6], [6, 5], [6, 7], [7, 6], [7, 8], [8, 7], [8, 9], [9, 8], [9, 10], [10, 9], [10, 11], [11, 10], [11, 12], [12, 11], [12, 13], [13, 12], [13, 14], [14, 13], [14, 15]]
####
for i in range(1, len(word_index) - 1):
    # (context, target) : ([target index - 1, target index + 1], target)
    target = word_index[i]
    context = [word_index[i - 1], word_index[i + 1]]

    # (target, context[0]), (target, context[1])..
    for w in context:
        skip_grams.append([target, w])
####
def random_batch(data, size):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace=False)

    for i in random_index:
        random_inputs.append(data[i][0])  # target
        random_labels.append([data[i][1]])  # context word

    return random_inputs, random_labels

#Input (batchsize * )target index(sentences->word_list->word_index->skip_grams_targetindex)
#Hidden0.0 embedding:len(word_list)*dimension(word_vec)
#Hidden0.1 selected:len(word_list)*dimension(word_vec)    
#Hidden1 len(word_list)*demension(word_vec)
#Output (batchsize * )context index(sentences->word_list->word_index->skip_grams_contextindex)
####
#input:index
#hidden:(index->(x,y)->index)embedding
#output:index    
####
print("that is, here training data,skip_gram, is the most important.\
       \nBut here, skip_gram is only based on wordlist extracted from sentences.\
       \nnot reliable")
####    
training_epoch = 300
learning_rate = 0.1
batch_size = 20
embedding_size = 2
num_sampled = 15
voc_size = len(word_list)
####
inputs = tf.placeholder(tf.int32, shape=[batch_size])
labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
selected_embed = tf.nn.embedding_lookup(embeddings, inputs)
nce_weights = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
nce_biases = tf.Variable(tf.zeros([voc_size]))
loss = tf.reduce_mean(
            tf.nn.nce_loss(nce_weights, nce_biases, labels, selected_embed, num_sampled, voc_size))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
print("\n5.4 Embedding Neural Network(Word2Vec)")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1, training_epoch + 1):
        batch_inputs, batch_labels = random_batch(skip_grams, batch_size)

        _, loss_val = sess.run([train_op, loss],
                               feed_dict={inputs: batch_inputs,
                                          labels: batch_labels})

        if step % 10 == 0:
            print("loss at step ", step, ": ", loss_val)

    trained_embeddings = embeddings.eval()
####
for i, label in enumerate(word_list):
    x, y = trained_embeddings[i]
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2),
                 textcoords='offset points', ha='right', va='bottom')
####
plt.show()
