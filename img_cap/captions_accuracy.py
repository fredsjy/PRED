import numpy as np
import tensorflow as tf
import pickle

# import word2vec_tf_idf as wv


# load datasets
# n = norm_dataset.NormData(captions_directory="../data/pascal-sentences/ps_captions/"
#                           ,label_file="../data/pascal-sentences/labels.txt")
#                           #,images_directory="../data/pascal-sentences/ps_images/")

# n = pickle.load(open("../captions_vectors_nn/normData", "rb"))
n = pickle.load(open("normData","rb"))
######### use word2vec
# vec = wv.docvec()
# train_data_nd = vec



######### use doc2vec
train_data_nd = np.array(n.captions)

train_label_nd = np.array(n.labels)
# test_data_nd = np.array(test_data)
# test_label_nd = np.array(test_label)

data_nd = np.array(n.captions)
label_nd = np.array(n.labels)

train_data = []
test_data = []
train_label = []
test_label = []
for i in range(1000):
    if i % 10 < 10:
        train_data.append(data_nd[i])
        train_label.append(label_nd[i])
    else:
        test_data.append(data_nd[i])
        test_label.append(label_nd[i])

# train_data_nd = np.array(train_data)
# train_label_nd = np.array(train_label)
test_data_nd = np.array(test_data)
test_label_nd = np.array(test_label)
#########



####### convert to tensor
# def my_func(arg):
#     arg = tf.convert_to_tensor(arg, dtype=tf.float32)
#     return arg

# train_data_tensor = my_func(train_data_nd)
# train_label_tensor = my_func(train_label_nd)

# test_data_tensor = my_func(test_data_nd)
# test_label_tensor = my_func(test_label_nd)
#########

# implement the regression
captions_dim = 1024
x = tf.placeholder(tf.float32,[None,captions_dim])

# layers
# def layer(input,input_dim,output_dim):
#     W = tf.Variable(tf.zeros([input_dim,output_dim]))
#     b = tf.Variable(tf.zeros([output_dim]))
#     y = tf.matmul(input, W) + b
#     param = namedtuple('parameter_of_layer','weight bias output')
#     return param(W,b,y)
# layer 1
W1 = tf.Variable(tf.zeros([1024, 20]))
b1 = tf.Variable(tf.zeros([20]))
y1 = tf.matmul(x, W1) + b1


# layer 2
# W2 = tf.Variable(tf.zeros([500, 300]))
# b2 = tf.Variable(tf.zeros([300]))
# y2 = tf.matmul(y1,W2)+b2
#
#
# # layer 3
# W3 = tf.Variable(tf.zeros([300, 20]))
# b3 = tf.Variable(tf.zeros([20]))
# y3 = tf.matmul(y2,W3)+b3

# # layer 4
# layer4 = layer(layer3.output,500,300)
#
# # layer 5
# layer5 = layer(layer4.output,300,100)
#
# # layer 6
# layer6 = layer(layer5.output,100,20)


# output
y = y1
y_ = tf.placeholder(tf.float32,[None,20])

# cross_entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for i in range(20000):
    sess.run(train_step, feed_dict={x: train_data_nd, y_: train_label_nd})
    if i % 200 == 0:
        print(i, ',accuracy:', sess.run(accuracy, feed_dict={x: train_data_nd, y_: train_label_nd}))


cap_vec = sess.run(tf.nn.softmax(y),feed_dict={x:train_data_nd})
pickle.dump(cap_vec,open('cap_vec','wb'))


