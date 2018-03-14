"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys,os
import tempfile
import gzip
import numpy

from subints import *
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
#from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import gfile
import tensorflow as tf

FLAGS = None

def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images(f):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth].
  Args:
    f: A file object that can be passed into a gzip reader.
  Returns:
    data: A 4D uint8 numpy array [index, y, x, depth].
  Raises:
    ValueError: If the bytestream does not start with 2051.
  """
  if f == "train":
    D_posi = numpy_array_stick("/home/luzihao/xiaoluo/dataset/bin/p309p_pfd",1,64)
    D_nega = numpy_array_stick("/home/luzihao/xiaoluo/dataset/bin/p309n_pfd",0,64)#use vstack to reconstruct the multi_array
    #"X" means data,"Y" means label
    Y_nega = D_nega[1]
    X_nega = D_nega[0]
    Y_posi = D_posi[1]
    X_posi = D_posi[0]
    temp_X = np.vstack((X_nega,X_posi))
    data_size = (np.shape(temp_X))[0]
    X = temp_X.reshape(data_size,64,64,1)
    Y = np.vstack((Y_nega,Y_posi))
    print("Train:")
    print(np.shape(X))
    print(np.shape(Y))
    return X,Y
  elif f == "test":
    D_posi = numpy_array_stick("/home/luzihao/xiaoluo/dataset/bin/Good_check_True",1,256)
    D_nega = numpy_array_stick("/home/luzihao/xiaoluo/dataset/bin/RFI_all_info",0,256)#use vstack to reconstruct the multi_array
    #"X" means data,"Y" means label
    Y_nega = D_nega[1]
    X_nega = D_nega[0]
    Y_posi = D_posi[1]
    X_posi = D_posi[0]
    temp_X = np.vstack((X_nega,X_posi))
    data_size = (np.shape(temp_X))[0]
    X = temp_X.reshape(data_size,64,64,1)
    Y = np.vstack((Y_nega,Y_posi))
    print("Test:")
    print(np.shape(X))
    print(np.shape(Y))
    return X,Y


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  print(numpy.shape(labels_one_hot))
  return labels_one_hot

# def extract_labels(f):
#   """Extract the labels into a 1D uint8 numpy array [index].
#   Args:
#     f: A file object that can be passed into a gzip reader.
#     one_hot: Does one hot encoding for the result.
#     num_classes: Number of classes for the one hot encoding.
#   Returns:
#     labels: a 1D uint8 numpy array.
#   Raises:
#     ValueError: If the bystream doesn't start with 2049.
#   """
#   if f == "train":
#     D_posi = numpy_array_stick("/home/luzihao/xiaoluo/dataset/bin/p309p_pfd",1,64)
#     D_nega = numpy_array_stick("/home/luzihao/xiaoluo/dataset/bin/p309n_pfd",0,64)#use vstack to reconstruct the multi_array
#     Y_nega = D_nega[1]#"X" means data,"Y" means label
#     Y_posi = D_posi[1]
#     Y = np.vstack((Y_nega,Y_posi))
#     print(np.shape(Y))
#     return Y
#   else:
#     D_posi = numpy_array_stick("/home/luzihao/xiaoluo/dataset/bin/Good_check_True",1,256)
#     D_nega = numpy_array_stick("/home/luzihao/xiaoluo/dataset/bin/RFI_all_info",0,256)
#     Y_nega = D_nega[1]
#     Y_posi = D_posi[1]
#     print(np.shape(Y))
#     return Y
#   #return labels

class DataSet(object):

  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    numpy.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 1000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      #fake_image = [1] * 784
      fake_image = [1]*4096
      if self.one_hot:
        #fake_label = [1] + [0] * 9
        fake_label = [1]+[0]*1
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]

  def read_data_sets(train_dir,
                     fake_data=False,
                     one_hot=False,
                     dtype=dtypes.float32,
                     reshape=True,
                     validation_size=200,
                     seed=None):
    if fake_data:

      def fake():
        return DataSet(
            [], [], fake_data=True, one_hot=one_hot, dtype=dtype, seed=seed)

      train = fake()
      validation = fake()
      test = fake()
      return base.Datasets(train=train, validation=validation, test=test)
    f1 = "train"
    train_images,train_labels = extract_images(f1)
    f2 = "test"
    test_images,test_labels = extract_images(f2)

    if not 0 <= validation_size <= len(train_images):
      raise ValueError(
          'Validation size should be between 0 and {}. Received: {}.'
          .format(len(train_images), validation_size))

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]


    options = dict(dtype=dtype, reshape=reshape, seed=seed)

    train = DataSet(train_images, train_labels, **options)
    validation = DataSet(validation_images, validation_labels, **options)
    test = DataSet(test_images, test_labels, **options)

    return base.Datasets(train=train, validation=validation, test=test)

def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 64, 64, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5,5,1,32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 4X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_4x4(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_4x4(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 64x64 image
  # is down to 4x4x64 feature maps -- maps this to 512 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([4 * 4 * 64, 512])
    b_fc1 = bias_variable([512])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 4*4*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 512 features to 2 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([512, 2])
    b_fc2 = bias_variable([2])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def max_pool_4x4(x):
  """max_pool_4x4 downsamples a feature map by 4X."""
  return tf.nn.max_pool(x, ksize=[1, 4, 4, 1],
                        strides=[1, 4, 4, 1], padding='SAME')

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main(_):
  # Import data
  mnist = DataSet.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 4096])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 2])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
      batch = mnist.train.next_batch(5)
      if i % 200 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
