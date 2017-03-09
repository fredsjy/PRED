import pickle
import random
import numpy as np
import tensorflow as tf

<<<<<<< HEAD
#
# n = norm_dataset.NormData(images_directory="../data/pascal-sentences/ps_images/"
#                           ,label_file="../data/pascal-sentences/labels.txt"
#                           ,captions_directory="../data/pascal-sentences/ps_captions/")
# pickle.dump(n,open('normData','wb'))

n = pickle.load(open("normData","rb"))
=======
n = pickle.load(open("../captions_vectors_nn/normData","rb"))
>>>>>>> cf5762eb4d1b65315256ebf32b0f792aa4575610

data_nd = np.array(n.images)
label_nd = np.array(n.labels)

train_data = []
test_data = []
train_label = []
test_label = []
for i in range(1000):
    if i % 10 < 6:
        train_data.append(data_nd[i])
        train_label.append(label_nd[i])
    else:
        test_data.append(data_nd[i])
        test_label.append(label_nd[i])

train_image = np.array(train_data)
train_label = np.array(train_label)
test_image = np.array(test_data)
test_label = np.array(test_label)

def shuffle_dataset(images, labels):
    image_shuffle = []
    label_shuffle = []
    container = []
    for i in range(len(images)):
        container.append((images[i],labels[i]))
    random.shuffle(container)
    for i in range(len(images)):
        x,y= container[i]
        image_shuffle.append(x)
        label_shuffle.append(y)
    image_shuffle = np.array(image_shuffle, np.float32)
    label_shuffle = np.array(label_shuffle, np.float32)
    return image_shuffle, label_shuffle

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 28,28,3])  # 28x28
ys = tf.placeholder(tf.float32, [None, 20])
keep_prob = tf.placeholder(tf.float32)

x_image = xs

## conv1 layer ##
W_conv1 = weight_variable([5, 5, 3, 16])  # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([16])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)    # output size 32个56x56
h_pool1 = max_pool_2x2(h_conv1)  # output size 32个14x14

## conv2 layer ##
W_conv2 = weight_variable([5, 5, 16, 32])  # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)  # output size 64个7x7

## if use fc1, use like this. if use fc2, replace the following code with "fc1 layer and fc2 layer"##
# ## fc1 layer ##
# h_pool2_flat = tf.reshape(h_pool2, [-1,32*7*7])
# W_fc1 = weight_variable([32*7*7, 20])
# b_fc1 = bias_variable([20])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# prediction = tf.nn.softmax(h_fc1)


## fc1 layer ##
h_pool2_flat = tf.reshape(h_pool2, [-1,32*7*7])
W_fc1 = weight_variable([32*7*7, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([1024, 20])
b_fc2 = bias_variable([20])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))  # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
if int((tf.__version__).split('.')[1]) < 12:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

<<<<<<< HEAD
for i in range(1000):
    # train_image_shuffle, train_label_shuffle = shuffle_dataset(train_image, train_label)
    # test_image_shuffle, test_label_shuffle = shuffle_dataset(test_image, test_label)

    # BATCH_SIZE = len(train_label_shuffle)
    # train_image_batch = np.array([None] * int(len(train_image_shuffle)/BATCH_SIZE))
    # train_label_batch = np.array([None] * int(len(train_label_shuffle)/BATCH_SIZE))
    # for i in range(int(len(train_image)/BATCH_SIZE)):
    #     train_image_batch[i] = train_image[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
    #     train_label_batch[i] = train_label[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
    # # train_image_batch1 = train_image[0:200]
    # # train_image_batch2 = train_image[200:400]
    # # train_image_batch3 = train_image[400:600]
    # # train_image_batch4 = train_image[600:800]
    # # train_image_batch5 = train_image[800:1000]
    # # train_label_batch1 = train_label[0:200]
    # # train_label_batch2 = train_label[200:400]
    # # train_label_batch3 = train_label[400:600]
    # # train_label_batch4 = train_label[600:800]
    # # train_label_batch5 = train_label[800:1000]
    #
    # # for i in range(30):
    # for ii in range(int(len(train_image_batch))):
=======
for i in range(500):
>>>>>>> cf5762eb4d1b65315256ebf32b0f792aa4575610
    sess.run(train_step,feed_dict={xs: data_nd,ys: label_nd, keep_prob: 0.5})
    if i % 10 == 0:
        accuracy = 100 * compute_accuracy(data_nd, label_nd)
<<<<<<< HEAD
        print(accuracy)

# vecs = sess.run(h_fc1_drop,feed_dict={xs:data_nd,keep_prob:0.5})

# get the vectors of fc2
fc2_pre = sess.run(prediction,feed_dict={xs:data_nd,keep_prob:0.5})
pickle.dump(fc2_pre,open('fc2_pre','wb'))

# get the vectors of fc1
# fc1_pre = sess.run(prediction,feed_dict={xs:data_nd,keep_prob:0.5})
# pickle.dump(fc1_pre,open('fc1_pre','wb'))
=======
        print(accuracy)
>>>>>>> cf5762eb4d1b65315256ebf32b0f792aa4575610
