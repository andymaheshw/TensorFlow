# Databricks notebook source exported at Thu, 17 Nov 2016 19:32:52 UTC
# MAGIC %md Deep Learning
# MAGIC =============
# MAGIC 
# MAGIC Assignment 6 | Tianzi Cai | 2016-11-06
# MAGIC 
# MAGIC ------------
# MAGIC 
# MAGIC After training a skip-gram model in 5_word2vec.ipynb, the goal of this notebook is to train a LSTM character model over [Text8](http://mattmahoney.net/dc/textdata) data.

# COMMAND ----------

try:
  import tensorflow as tf
  print("TensorFlow is already installed")
except ImportError:
  print("Installing TensorFlow")
  import subprocess
  subprocess.check_call(["/databricks/python/bin/pip", "install", "https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0-cp27-none-linux_x86_64.whl"])
  
  print("TensorFlow has been installed on this cluster")

# COMMAND ----------

from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve
import itertools

# COMMAND ----------

# MAGIC %md Download the data from the source website if necessary.

# COMMAND ----------

url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified %s' % filename)
  else:
    print(statinfo.st_size)
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

filename = maybe_download('text8.zip', 31344016)

# COMMAND ----------

# MAGIC %md Read the data into a string.

# COMMAND ----------

def read_data(filename):
  f = zipfile.ZipFile(filename)
  for name in f.namelist():
    return tf.compat.as_str(f.read(name))
  f.close()
  
text = read_data(filename)
print('Data size %d' % len(text))

# COMMAND ----------

# MAGIC %md Create a small validation set. 

# COMMAND ----------

valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)
print(train_size, train_text[:64])
print(valid_size, valid_text[:64])

# COMMAND ----------

# MAGIC %md Utility functions to map characters to vocabulary IDs and back.

# COMMAND ----------

vocabulary_size = len(string.ascii_lowercase) + 1 # [a-z] + ' ', or 27
first_letter = ord(string.ascii_lowercase[0])

def char2id(char):
  if char in string.ascii_lowercase:
    return ord(char) - first_letter + 1
  elif char == ' ':
    return 0
  else:
    print('Unexpected character: %s' % char)
    return 0
  
def id2char(dictid):
  if dictid > 0:
    return chr(dictid + first_letter - 1)
  else:
    return ' '

print(char2id('a'), char2id('z'), char2id(' '), char2id('ï'))
print(id2char(1), id2char(26), id2char(0))
# print(string.ascii_lowercase)
# print(string.ascii_letters)

# COMMAND ----------

# MAGIC %md Function to generate a training batch for the LSTM model.

# COMMAND ----------

batch_size=64
num_unrollings=10

class BatchGenerator(object):
  def __init__(self, text, batch_size, num_unrollings):
    '''properties of the class'''
    self._text = text
    self._text_size = len(text)
    self._batch_size = batch_size
    self._num_unrollings = num_unrollings
    segment = self._text_size // batch_size # 99999000/64 = 1562484
    self._cursor = [ offset * segment for offset in range(batch_size)] # [1562484, 14162484*2, 1562484*3, ... 1562484*63]
    self._last_batch = self._next_batch() 
  
  def _next_batch(self):
    """Generate a single batch from the current cursor position in the data."""
    '''Names, in a class, with a leading underscore are simply to indicate to 
    other programmers that the attribute or method is intended to be private.
    batch takes on the shape of 11 by 64 by 27'''
    batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float) # (64, 27)
    for b in range(self._batch_size):
      '''
      b = 0, ... 63
      self._cursor[b] = self._cursor[0] = 1562484
      self._text[self_cursor[0]] = self._text[1562484] = 'w'
      char2id(self._text[self._cursor[0]]) = 23
      self._cursor[0] = 1562484 + 1
      return batch of shape (64, 27) of all zeros except 1 in one unique position within each row
      '''
      batch[b, char2id(self._text[self._cursor[b]])] = 1.0
      self._cursor[b] = (self._cursor[b] + 1) % self._text_size
    return batch
  
  def next(self):
    """Generate the next array of batches from the data. The array consists of
    the last batch of the previous array, followed by num_unrollings new ones.
    """
    '''
    self._last_batch is initialized to be the first batch in the beginning
    but each time the function next() gets called, it called self._next_batch()
    and self._cursor gets undated with a new array of 64 numbers 
    and if the numbers start to get larger than self._text_size, 
    they are divided by self._text_size and we take the remainder
    this ensure that there are always usable numbers in the self._cursor for 
    self._text[self._cursor[any(num_unrollings)]] to find a valid character
    
    batches gets initialized to be the first batch the first time, the last batch all following times
    batches gets appended (num_unrolling=) 10 times, each time adding (64, 27) to it 
    batches become a list of 11, each a numpy array of shape (64, 27)
    cursor gets updated too each time
    self._last_batch becomes the last element in the batches list, that's where cursors were
    last pointing at ***
    '''
    batches = [self._last_batch]
    for step in range(self._num_unrollings):
      batches.append(self._next_batch()) 
    self._last_batch = batches[-1]
    return batches

def characters(probabilities):
  """Turn a 1-hot encoding or a probability distribution over the possible
  characters back into its (most likely) character representation."""
  return [id2char(c) for c in np.argmax(probabilities, 1)]

def batches2string(batches):
  """Convert a sequence of batches back into their *(most likely)* string
  representation."""
  '''2016-10-16 tz: zip() is where the strings get longer to the length of num_unrolling + 1
  The for loop could also be written as for b in num_unrolling + 1: '''
  s = [''] * batches[0].shape[0]
  for b in batches:
    s = [''.join(x) for x in zip(s, characters(b))]
  return s

train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
valid_batches = BatchGenerator(valid_text, 1, 1)

print('2 train_batches of length {}'.format(len(batches2string(train_batches.next()))))
print(batches2string(train_batches.next())[:3])
print(batches2string(train_batches.next())[:3])
print('2 valid_batches of length {}'.format(len(batches2string(valid_batches.next()))))
print(batches2string(valid_batches.next()))
print(batches2string(valid_batches.next()))

# COMMAND ----------

curious_next = train_batches.next()
print(len(curious_next))
print(curious_next[0].shape)
print(zip(characters(curious_next[0]), characters(curious_next[1]), 
          characters(curious_next[2]), characters(curious_next[3]), 
          characters(curious_next[4]), characters(curious_next[5]))[:3])

# COMMAND ----------

def logprob(predictions, labels):
  """Log-probability of the true labels in a predicted batch."""
  '''2016-10-16 tz: ? how is this formula construed?'''
  predictions[predictions < 1e-10] = 1e-10
  return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

def sample_distribution(distribution):
  """Sample one element from a distribution assumed to be an array of normalized
  probabilities.
  """
  '''2016-10-16 tz: return exits for loop'''
  '''2016-10-16 tz: why go through so much trouble to return a random int from 0 to 26?'''
  '''2016-10-18 tz: 
  ohhhhh, this function returns the position with a probability of position.value of the time
  AHHHHHH!!'''
  r = random.uniform(0, 1)
  s = 0
  for i in range(len(distribution)):
    s += distribution[i]
    if s >= r:
      return i
  return len(distribution) - 1

def sample(prediction):
  """Turn a (column) prediction into 1-hot encoded samples."""
  p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
  p[0, sample_distribution(prediction[0])] = 1.0
  return p

def random_distribution():
  """Generate a random column of probabilities."""
  '''why use [:, None]? without that part, everything would work too'''
  '''2016-10-16 tz: [:, None] makes sure that the np array can be broadcast for matrix division'''
  b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
  return b/np.sum(b, 1)[:,None]

# 2016-10-16 tz: testing code
a = random_distribution()
print('random_distribution of shape {}: {}\n'.format(a.shape, a))

b = sample_distribution(a[0])
print('sample_distribution() returns a random integer between 0 and 26: {}\n'.format(b))

feed = sample(a)
print('sample() returns a 1-hot encoded sample: {}\n'.format(feed))

sentence = characters(feed)[0]
print('characters(feed) returns: {}'.format(characters(feed)))
print('characters(feed)[0] returns: {}'.format(sentence))

# COMMAND ----------

# The sample distribution is weighted by the predicted probabilities. So, for example, the following code will print 2 with 97% probability:
dist = np.array([0.01, 0.01, 0.97, 0.01])
a = []
b = []
n = 10000
for i in range(n):
  a.append(sample_distribution(dist))
  b.append(np.random.choice([0,1,2,3], p=[0.01, 0.01, 0.97, 0.01]))
print('sample_distribution(dist) returns {:.1%} of the time'.format(a.count(2)/float(n)))
print('np.random.choice(dist) returns {:.1%} of the time'.format(b.count(2)/float(n)))

# COMMAND ----------

def logprob(predictions, labels):
  """Log-probability of the true labels in a predicted batch."""
  '''2016-10-16 tz: (see below)'''
  predictions[predictions < 1e-10] = 1e-10
  return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

p = np.array([[  1.16494857e-01,   5.48233045e-04,  1.89603455e-02,  2.41364352e-02,
    1.45250736e-02,   5.68647345e-04,   1.56472325e-02,   4.08084914e-02,
    1.86075253e-04,   8.71601701e-03,   2.35217012e-04,   4.18233965e-03,
    9.12578106e-02,   8.86521861e-03,   2.86392480e-01,   5.85887647e-05,
    8.62208102e-03,   3.18372448e-04,   1.69273183e-01,   1.35227934e-01,
    3.57363336e-02,   1.02764592e-02,   4.18751407e-03,   8.80928070e-04,
    3.22445121e-04,   7.79473106e-04,   2.79215141e-03], [  1.16494857e-01,   5.48233045e-04,  1.89603455e-02,  2.41364352e-02,
    1.45250736e-02,   5.68647345e-04,   1.56472325e-02,   4.08084914e-02,
    1.86075253e-04,   8.71601701e-03,   2.35217012e-04,   4.18233965e-03,
    9.12578106e-02,   8.86521861e-03,   2.86392480e-01,   5.85887647e-05,
    8.62208102e-03,   3.18372448e-04,   1.69273183e-01,   1.35227934e-01,
    3.57363336e-02,   1.02764592e-02,   4.18751407e-03,   8.80928070e-04,
    3.22445121e-04,   7.79473106e-04,   2.79215141e-03]])
l = np.array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
   0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.], [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

print('predicions of shape {}, first element looks like: {}'.format(p.shape, p[0]))
print('labels of same shape {}, first element looks like: {}\n'.format(l.shape, l[0]))
print('np.multiply(labels, -np.log(predictions)) does element-wise multiplication: \n{}\n'.format(np.multiply(l, -np.log(p))))
print('np.sum(np.multiply(labels, -np.log(predictions))) computes total perplexity: {}'.format(np.sum(np.multiply(l, -np.log(p)))))
print('average perpelxity across batch (2 here): {}'.format(logprob(predictions = p, labels = l)))

# COMMAND ----------

# MAGIC %md -np.log() makes sure that probabilities close to 0 are penalized more where as probabilities close to 1 contribute very little to perplexities. 

# COMMAND ----------

train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
valid_batches = BatchGenerator(valid_text, 1, 1)

num_nodes = 64

graph = tf.Graph()
with graph.as_default():
  
  # Parameters:
  # Input gate: input, previous output, and bias. # Is previous output also known as "cell output activation"
  ix = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1)) # 27, 64
  im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1)) # 64, 64
  ib = tf.Variable(tf.zeros([1, num_nodes])) # 1, 64
  # Forget gate: input, previous output, and bias.
  fx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
  fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  fb = tf.Variable(tf.zeros([1, num_nodes]))
  # Memory cell: input, state and bias.                             
  cx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
  cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  cb = tf.Variable(tf.zeros([1, num_nodes]))
  # Output gate: input, previous output, and bias.
  ox = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
  om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  ob = tf.Variable(tf.zeros([1, num_nodes]))
  # Variables saving state across unrollings.
  '''trainable: If True, the default, also adds the variable to the graph collection 
  GraphKeys.TRAINABLE_VARIABLES. 
  This collection is used as the default list of variables to use by the Optimizer classes.'''
  saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False) # 64, 64
  saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False) # 64, 64
  # Classifier weights and biases.
  w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1)) # 64, 27
  b = tf.Variable(tf.zeros([vocabulary_size])) # 27
  
  # Definition of the cell computation.
  def lstm_cell(i, o, state):
    """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
    Note that in this formulation, we omit the various connections between the
    previous state and the gates."""
    input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
    forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
    update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
    state = forget_gate * state + input_gate * tf.tanh(update)
    output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
    return output_gate * tf.tanh(state), state

  # Input data.
  train_data = list()
  for _ in range(num_unrollings + 1):
    train_data.append(tf.placeholder(tf.float32, shape=[batch_size,vocabulary_size]))
  train_inputs = train_data[:num_unrollings]
  train_labels = train_data[1:]  # labels are inputs shifted by one time step.

  # Unrolled LSTM loop.
  outputs = list()
  output = saved_output
  state = saved_state
  for i in train_inputs:
    output, state = lstm_cell(i, output, state)
    outputs.append(output)

  # State saving across unrollings.
  # `logits` and `loss` will only run after `save_output` and `saved_state` 
  # have been assigned. 
  with tf.control_dependencies([saved_output.assign(output),
                                saved_state.assign(state)]):
    # Classifier.
    logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
    loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits, tf.concat(0, train_labels)))

  # Optimizer.
  global_step = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(10.0, global_step, 5000, 0.1, staircase=True)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  gradients, v = zip(*optimizer.compute_gradients(loss))
  gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
  optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

  # Predictions.
  # train_prediction has shape (64 * 10, 27). 
  train_prediction = tf.nn.softmax(logits) 
  
  # Sampling and validation eval: batch 1, no unrolling.
  sample_input = tf.placeholder(tf.float32, shape=[1, vocabulary_size])
  saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
  saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
  reset_sample_state = tf.group(
    saved_sample_output.assign(tf.zeros([1, num_nodes])),
    saved_sample_state.assign(tf.zeros([1, num_nodes])))
  sample_output, sample_state = lstm_cell(
    sample_input, saved_sample_output, saved_sample_state)
  # `sample_prediction` will only run when `saved_sample_output`
  # and `saved_sample_state` have been assigned
  with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                saved_sample_state.assign(sample_state)]):
    sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))

# COMMAND ----------

with tf.Session(graph=graph) as session:
  batches = train_batches.next()
  feed_dict = dict()
  for i in range(11):
    feed_dict[train_data[i]] = batches[i]
  tf.initialize_all_variables().run()
  x, y = session.run([logits, train_prediction], feed_dict = feed_dict)
  print(x[0])
  print(y[0])
  print(x.shape)
  print(y.shape)

# COMMAND ----------

fake_feed_dict = dict()
fake_train_data = list()
# IndexError: list index out of range
fake_feed_dict[fake_train_data[0]] = curious_next[0] 
print(type(curious_next[0]))

# Where as this works, because `train_data` is a list of tensors.
# even if it is initialized in a graph the same way
fake_feed_dict[train_data[0]] = curious_next[0]
print(type(train_data[0]))

# COMMAND ----------

num_steps = 7001
summary_frequency = 100

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  mean_loss = 0
  for step in range(num_steps):
    batches = train_batches.next()
    feed_dict = dict()
    for i in range(num_unrollings + 1):
      feed_dict[train_data[i]] = batches[i]
      
    _, l, predictions, lr = session.run([optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
    mean_loss += l
    if step % summary_frequency == 0:
      if step > 0:
        mean_loss = mean_loss / summary_frequency
      # The mean loss is an estimate of the loss over the last few batches.
      print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
      mean_loss = 0
      labels = np.concatenate(list(batches)[1:])
      print('Minibatch perplexity: %.2f' % float(np.exp(logprob(predictions, labels))))

      if step % (summary_frequency * 10) == 0:
        # Generate some samples.
        print('=' * 80)
        for _ in range(5):
          # tz: random letter generation
          feed = sample(random_distribution())
          sentence = characters(feed)[0]
          reset_sample_state.run()
          for _ in range(79):
            prediction = sample_prediction.eval({sample_input: feed})
            feed = sample(prediction) #!!!!
            sentence += characters(feed)[0]
          print(sentence)
        print('=' * 80)
        
      # Measure validation set perplexity.
      reset_sample_state.run()
      valid_logprob = 0
      for _ in range(valid_size): # tz: valid_size = 1000
        b = valid_batches.next()
        predictions = sample_prediction.eval({sample_input: b[0]})
        valid_logprob = valid_logprob + logprob(predictions, b[1])
      print('Validation set perplexity: %.2f' % float(np.exp(valid_logprob / valid_size)))

# COMMAND ----------

print('train_data is a list of length {}. train_data[0] has shape {}\n'.format(len(train_data), train_data[0].get_shape()))
print('train_inputs is a list of length {}. train_inputs[0] has shape {}\n'.format(len(train_inputs), train_inputs[0].get_shape()))
print('train_labels is a list of length {}. train_labels[0] has shape {}\n'.format(len(train_labels), train_labels[0].get_shape()))

# COMMAND ----------

# MAGIC %md
# MAGIC You might have noticed that the definition of the LSTM cell involves 4 matrix multiplications with the input, and 4 matrix multiplications with the output. Simplify the expression by using a single matrix multiply for each, and variables that are 4 times larger.

# COMMAND ----------

num_nodes = 64

graph = tf.Graph()
with graph.as_default():
  
  # Parameters:
  # Input/Forget/Update/Output gate: input, previous output, and bias.  
  ifcox = tf.Variable(tf.truncated_normal([vocabulary_size, 4 * num_nodes], -0.1, 0.1))
  ifcom = tf.Variable(tf.truncated_normal([num_nodes, 4 * num_nodes], -0.1, 0.1))
  ifcob = tf.Variable(tf.zeros([1, 4 * num_nodes]))
  
  # Variables saving state across unrollings.
  saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  # Classifier weights and biases.
  w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
  b = tf.Variable(tf.zeros([vocabulary_size]))
  
  # Definition of the cell computation.
  def lstm_cell(i, o, state):
    all_gates = tf.matmul(i, ifcox) + tf.matmul(o, ifcom) + ifcob
    input_gate = tf.sigmoid(all_gates[:, :num_nodes])
    forget_gate = tf.sigmoid(all_gates[:, num_nodes:2*num_nodes])
    update = tf.sigmoid(all_gates[:, 2*num_nodes:3*num_nodes])
    state = forget_gate * state + input_gate * tf.tanh(update)
    output_gate = tf.sigmoid(all_gates[:, 3*num_nodes:])
    return output_gate * tf.tanh(state), state

  # Input data.
  train_data = list()
  for _ in range(num_unrollings + 1):
    train_data.append(tf.placeholder(tf.float32, shape=[batch_size,vocabulary_size]))
  train_inputs = train_data[:num_unrollings]
  train_labels = train_data[1:]  # labels are inputs shifted by one time step.
  train_data = train_data * 1
  
  # Unrolled LSTM loop.
  outputs = list()
  output = saved_output
  state = saved_state
  for i in train_inputs:
    output, state = lstm_cell(i, output, state)
    outputs.append(output)

  # State saving across unrollings.
  with tf.control_dependencies([saved_output.assign(output),
                                saved_state.assign(state)]):
    # Classifier.
    logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
    loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits, tf.concat(0, train_labels)))

  # Optimizer.
  global_step = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(10.0, global_step, 5000, 0.1, staircase=True)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  gradients, v = zip(*optimizer.compute_gradients(loss))
  gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
  optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

  # Predictions.
  # 2016-10-16 tz: 640, 27
  train_prediction = tf.nn.softmax(logits) 
  
  # Sampling and validation eval: batch 1, no unrolling.
  sample_input = tf.placeholder(tf.float32, shape=[1, vocabulary_size])
  saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
  saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
  # tz note: 
  reset_sample_state = tf.group(
    saved_sample_output.assign(tf.zeros([1, num_nodes])),
    saved_sample_state.assign(tf.zeros([1, num_nodes])))
  sample_output, sample_state = lstm_cell(
    sample_input, saved_sample_output, saved_sample_state)
  with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                saved_sample_state.assign(sample_state)]):
    sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b)) # tz: ok, this is where sample_prediction is. 

# COMMAND ----------

num_steps = 7001
summary_frequency = 100

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  mean_loss = 0
  for step in range(num_steps):
    batches = train_batches.next()
    feed_dict = dict()
    for i in range(num_unrollings + 1):
      feed_dict[train_data[i]] = batches[i]
      
    # 2016-10-16 tz: session.run() evaluates same variables in a list
    _, l, predictions, lr = session.run([optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
    mean_loss += l
    if step % summary_frequency == 0:
      if step > 0:
        mean_loss = mean_loss / summary_frequency
      # The mean loss is an estimate of the loss over the last few batches.
      print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
      mean_loss = 0
      labels = np.concatenate(list(batches)[1:])
      print('Minibatch perplexity: %.2f' % float(np.exp(logprob(predictions, labels))))

      if step % (summary_frequency * 10) == 0:
        # Generate some samples.
        print('=' * 80)
        for _ in range(5):
          # tz: random letter generation
          feed = sample(random_distribution())
          sentence = characters(feed)[0]
          reset_sample_state.run()
          for _ in range(79):
            prediction = sample_prediction.eval({sample_input: feed})
            feed = sample(prediction)
            sentence += characters(feed)[0]
          print(sentence)
        print('=' * 80)
        
      # Measure validation set perplexity.
      reset_sample_state.run()
      valid_logprob = 0
      for _ in range(valid_size): # tz: valid_size = 1000
        b = valid_batches.next()
        predictions = sample_prediction.eval({sample_input: b[0]})
        valid_logprob = valid_logprob + logprob(predictions, b[1])
      print('Validation set perplexity: %.2f' % float(np.exp(valid_logprob / valid_size)))

# COMMAND ----------

num_nodes = 64

graph = tf.Graph()
with graph.as_default():
  
  # Parameters:
  # Input/Forget/Update/Output gate: input, previous output, and bias.  
  ifcob = tf.Variable(tf.zeros([1, 4 * num_nodes]))
  ifcoxm = tf.Variable(tf.truncated_normal([vocabulary_size + num_nodes, 4 * num_nodes], -0.1, 0.1))
  
  # Variables saving state across unrollings.
  saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  # Classifier weights and biases.
  w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
  b = tf.Variable(tf.zeros([vocabulary_size]))
  
  # Definition of the cell computation.
  def lstm_cell(i, o, state):
    all_gates = tf.matmul(tf.concat(1, [i, o]), ifcoxm) + ifcob
    input_gate = tf.sigmoid(all_gates[:, :num_nodes])
    forget_gate = tf.sigmoid(all_gates[:, num_nodes:2*num_nodes])
    update = tf.sigmoid(all_gates[:, 2*num_nodes:3*num_nodes])
    state = forget_gate * state + input_gate * tf.tanh(update)
    output_gate = tf.sigmoid(all_gates[:, 3*num_nodes:])
    return output_gate * tf.tanh(state), state

  # Input data.
  train_data = list()
  for _ in range(num_unrollings + 1):
    train_data.append(tf.placeholder(tf.float32, shape=[batch_size,vocabulary_size]))
  train_inputs = train_data[:num_unrollings]
  train_labels = train_data[1:]  # labels are inputs shifted by one time step.

  # Unrolled LSTM loop.
  outputs = list()
  output = saved_output
  state = saved_state
  for i in train_inputs:
    # print(tf.argmax(i, 1).get_shape())
    # print(tf.nn.embedding_lookup(embed, tf.argmax(i, 1)).get_shape()) # TensorShape([Dimension(64), Dimension(256)])
    output, state = lstm_cell(i, output, state)
    outputs.append(output)

  # State saving across unrollings.
  with tf.control_dependencies([saved_output.assign(output),
                                saved_state.assign(state)]):
    # Classifier.
    logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
    loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits, tf.concat(0, train_labels)))

  # Optimizer.
  global_step = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(10.0, global_step, 5000, 0.1, staircase=True)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  gradients, v = zip(*optimizer.compute_gradients(loss))
  gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
  optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

  # Predictions.
  # 2016-10-16 tz: 640, 27
  train_prediction = tf.nn.softmax(logits) 
  
  # Sampling and validation eval: batch 1, no unrolling.
  sample_input = tf.placeholder(tf.float32, shape=[1, vocabulary_size])
  saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
  saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
  # tz note: 
  reset_sample_state = tf.group(
    saved_sample_output.assign(tf.zeros([1, num_nodes])),
    saved_sample_state.assign(tf.zeros([1, num_nodes])))
  sample_output, sample_state = lstm_cell(
    sample_input, saved_sample_output, saved_sample_state)
  with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                saved_sample_state.assign(sample_state)]):
    sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b)) # tz: ok, this is where sample_prediction is. 

# COMMAND ----------

num_steps = 7001
summary_frequency = 100

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  mean_loss = 0
  for step in range(num_steps):
    batches = train_batches.next()
    feed_dict = dict()
    for i in range(num_unrollings + 1):
      feed_dict[train_data[i]] = batches[i]
      
    # 2016-10-16 tz: session.run() evaluates same variables in a list
    _, l, predictions, lr = session.run([optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
    mean_loss += l
    if step % summary_frequency == 0:
      if step > 0:
        mean_loss = mean_loss / summary_frequency
      # The mean loss is an estimate of the loss over the last few batches.
      print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
      mean_loss = 0
      labels = np.concatenate(list(batches)[1:])
      print('Minibatch perplexity: %.2f' % float(np.exp(logprob(predictions, labels))))

      if step % (summary_frequency * 10) == 0:
        # Generate some samples.
        print('=' * 80)
        for _ in range(5):
          # tz: random letter generation
          feed = sample(random_distribution())
          sentence = characters(feed)[0]
          reset_sample_state.run()
          for _ in range(79):
            prediction = sample_prediction.eval({sample_input: feed})
            feed = sample(prediction)
            sentence += characters(feed)[0]
          print(sentence)
        print('=' * 80)
        
      # Measure validation set perplexity.
      reset_sample_state.run()
      valid_logprob = 0
      for _ in range(valid_size): # tz: valid_size = 1000
        b = valid_batches.next()
        predictions = sample_prediction.eval({sample_input: b[0]})
        valid_logprob = valid_logprob + logprob(predictions, b[1])
      print('Validation set perplexity: %.2f' % float(np.exp(valid_logprob / valid_size)))

# COMMAND ----------

# MAGIC %md
# MAGIC We want to train a LSTM over bigrams, that is pairs of consecutive characters like 'ab' instead of single characters like 'a'. Since the number of possible bigrams is large, feeding them directly to the LSTM using 1-hot encodings will lead to a very sparse representation that is very wasteful computationally.
# MAGIC 
# MAGIC a- Introduce an embedding lookup on the inputs, and feed the embeddings to the LSTM cell instead of the inputs themselves.
# MAGIC 
# MAGIC b- Write a bigram-based LSTM, modeled on the character LSTM above.
# MAGIC 
# MAGIC c- Introduce Dropout. For best practices on how to use Dropout in LSTMs, refer to [this article](http://arxiv.org/abs/1409.2329).

# COMMAND ----------

bigrams_vocab = [x+y for x in string.ascii_lowercase + ' ' for y in string.ascii_lowercase + ' ']
bigrams_embed = np.eye(len(bigrams_vocab))

def createBigramsTrainTest(train_data):
  train_inputs = list()
  for k in range(num_unrollings-1):
    bigrams_pos = list()
    for i in range(batch_size):
      bigram = id2char(np.argmax(train_data[k][i,:])) + id2char(np.argmax(train_data[k+1][i,:]))
      bigrams_pos.append(bigrams_vocab.index(bigram))
    train_inputs.append(bigrams_embed[bigrams_pos,:])
  train_labels = train_data[2:]
  return train_inputs, train_labels

num_unrollings = 11
train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
valid_batches = BatchGenerator(valid_text, 1, 1)

test_train_data = train_batches.next()
train_inputs, train_labels = createBigramsTrainTest(test_train_data)

print(len(train_inputs))
print((train_inputs[0]).shape)
print(len(train_labels))
print((train_labels[0]).shape)

nr_pos = 2
nb_pos = 2
print(np.argmax(train_inputs[nr_pos][nb_pos,:]))
print(bigrams_vocab[np.argmax(train_inputs[nr_pos][nb_pos,:])])
print(np.argmax(train_labels[nr_pos][nb_pos,:]))
print(id2char(np.argmax(train_labels[nr_pos][nb_pos,:])))

sess = tf.Session()
print(tf.argmax(test_train_data[0], 1).eval(session=sess))
print(tf.nn.embedding_lookup(bigrams_embed, [1, 2]).eval(session=sess))
print(tf.argmax(train_inputs[0], 1).eval(session=sess))

# COMMAND ----------

num_nodes = 64

graph = tf.Graph()
with graph.as_default():
  
  # Parameters:
  # Input/Forget/Update/Output gate: input, previous output, and bias.  
  ifcom = tf.Variable(tf.truncated_normal([num_nodes, 4 * num_nodes], -0.1, 0.1))
  ifcob = tf.Variable(tf.zeros([1, 4 * num_nodes]))
  embeddings = tf.Variable(tf.truncated_normal([vocabulary_size, 4 * num_nodes], -0.1, 0.1))
  
  # Variables saving state across unrollings.
  saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  # Classifier weights and biases.
  w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
  b = tf.Variable(tf.zeros([vocabulary_size]))
  
  # Definition of the cell computation.
  def lstm_cell(i, o, state):
    all_gates = i + tf.matmul(o, ifcom) + ifcob
    input_gate = tf.sigmoid(all_gates[:, :num_nodes])
    forget_gate = tf.sigmoid(all_gates[:, num_nodes:2*num_nodes])
    update = tf.sigmoid(all_gates[:, 2*num_nodes:3*num_nodes])
    state = forget_gate * state + input_gate * tf.tanh(update)
    output_gate = tf.sigmoid(all_gates[:, 3*num_nodes:])
    return output_gate * tf.tanh(state), state

  # Input data.
  train_data = list()
  for _ in range(num_unrollings + 1):
    train_data.append(tf.placeholder(tf.float32, shape=[batch_size,vocabulary_size])) # changes made here
  train_inputs = train_data[:num_unrollings]
  train_labels = train_data[1:]  # labels are inputs shifted by one time step.
  train_data = train_data * 1
  
  # Unrolled LSTM loop.
  outputs = list()
  output = saved_output
  state = saved_state
  for i in train_inputs:
    idx = tf.argmax(i, 1)
    embedded = tf.nn.embedding_lookup(embeddings, idx)
    output, state = lstm_cell(embedded, output, state)
    outputs.append(output)
    
  # State saving across unrollings.
  with tf.control_dependencies([saved_output.assign(output),
                                saved_state.assign(state)]):
    # Classifier.
    logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
    loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits, tf.concat(0, train_labels)))

  # Optimizer.
  global_step = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(10.0, global_step, 5000, 0.1, staircase=True)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  gradients, v = zip(*optimizer.compute_gradients(loss))
  gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
  optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

  # Predictions.
  train_prediction = tf.nn.softmax(logits) 
  
  # Sampling and validation eval: batch 1, no unrolling.
  sample_input = tf.placeholder(tf.float32, shape=[1, vocabulary_size])
  # update sample_input for LTSM cell.
  embedded_sample_input = tf.nn.embedding_lookup(embeddings, tf.argmax(sample_input, 1))
  saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
  saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
  # tz note: 
  reset_sample_state = tf.group(
    saved_sample_output.assign(tf.zeros([1, num_nodes])),
    saved_sample_state.assign(tf.zeros([1, num_nodes])))
  sample_output, sample_state = lstm_cell(
    embedded_sample_input, saved_sample_output, saved_sample_state)
  with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                saved_sample_state.assign(sample_state)]):
    sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))

# COMMAND ----------

num_steps = 7001
summary_frequency = 100

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  mean_loss = 0
  for step in range(num_steps):
    batches = train_batches.next()
    feed_dict = dict()
    for i in range(num_unrollings + 1):
      feed_dict[train_data[i]] = batches[i]
      
    _, l, predictions, lr = session.run([optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
    mean_loss += l
    if step % summary_frequency == 0:
      if step > 0:
        mean_loss = mean_loss / summary_frequency
      # The mean loss is an estimate of the loss over the last few batches.
      print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
      mean_loss = 0
      labels = np.concatenate(list(batches)[1:])
      print('Minibatch perplexity: %.2f' % float(np.exp(logprob(predictions, labels))))

      if step % (summary_frequency * 10) == 0:
        # Generate some samples.
        print('=' * 80)
        for _ in range(5):
          # tz: random letter generation
          feed = sample(random_distribution())
          sentence = characters(feed)[0]
          reset_sample_state.run()
          for _ in range(79):
            prediction = sample_prediction.eval({sample_input: feed})
            feed = sample(prediction)
            sentence += characters(feed)[0]
          print(sentence)
        print('=' * 80)
        
      # Measure validation set perplexity.
      reset_sample_state.run()
      valid_logprob = 0
      for _ in range(valid_size): # tz: valid_size = 1000
        b = valid_batches.next()
        predictions = sample_prediction.eval({sample_input: b[0]})
        valid_logprob = valid_logprob + logprob(predictions, b[1])
      print('Validation set perplexity: %.2f' % float(np.exp(valid_logprob / valid_size)))

# COMMAND ----------

t1 = tf.Variable([1, 2, 3])
t2 = [7, 8, 11]
print(t1.get_shape())
print(tf.concat(0, [t1, t2]).get_shape())
t1 = [[1, 2, 3]]
t2 = [[7, 8, 11]]
print(tf.concat(1, [t1, t2]).get_shape())

x = list(batches2string(test_train_data)[0])
print(x)
y = list(zip(x[:-1:], x[1:-1:]))
print(y)
z = [''.join(i) for i in y]
print(z)

valid_batches = BatchGenerator(valid_text, 1, 2)
b = valid_batches.next()
print(b)
print(batches2string(b))
print(batches2string(b)[0])
print(batches2string(b)[0][-1])

# COMMAND ----------

num_nodes = 64
embedding_size = 128
num_grams = 2

graph = tf.Graph()
with graph.as_default():
  
  # Parameters:
  # Input/Forget/Update/Output gate: input, previous output, and bias.  
  ifcox = tf.Variable(tf.truncated_normal([embedding_size, 4 * num_nodes], -0.1, 0.1))
  ifcom = tf.Variable(tf.truncated_normal([num_nodes, 4 * num_nodes], -0.1, 0.1))
  ifcob = tf.Variable(tf.zeros([1, 4 * num_nodes]))
  embeddings = tf.Variable(tf.random_uniform([vocabulary_size * vocabulary_size, embedding_size], -1.0, 1.0))
  
  # Variables saving state across unrollings.
  saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  # Classifier weights and biases.
  w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
  b = tf.Variable(tf.zeros([vocabulary_size]))
  
  # Definition of the cell computation.
  def lstm_cell(i, o, state):
    all_gates = tf.matmul(i, ifcox) + tf.matmul(o, ifcom) + ifcob
    input_gate = tf.sigmoid(all_gates[:, :num_nodes])
    forget_gate = tf.sigmoid(all_gates[:, num_nodes:2*num_nodes])
    update = tf.sigmoid(all_gates[:, 2*num_nodes:3*num_nodes])
    state = forget_gate * state + input_gate * tf.tanh(update)
    output_gate = tf.sigmoid(all_gates[:, 3*num_nodes:])
    return output_gate * tf.tanh(state), state

  # Input data.
  train_data = list()
  for _ in range(num_unrollings + 1):
    train_data.append(tf.placeholder(tf.float32, shape=[batch_size,vocabulary_size]))
  train_inputs = zip(train_data[:-2:], train_data[1:-1:])
  train_labels = train_data[2:]
  
  # Unrolled LSTM loop.
  outputs = list()
  output = saved_output
  state = saved_state
  for i in train_inputs:
    ids = tf.argmax(i[0],1) * vocabulary_size + tf.argmax(i[1],1)
    embedded_i = tf.nn.embedding_lookup(embeddings, ids)
    output, state = lstm_cell(embedded_i, output, state)
    outputs.append(output)
  
  # State saving across unrollings.
  with tf.control_dependencies([saved_output.assign(output),
                                saved_state.assign(state)]):
    # Classifier.
    logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
    loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits, tf.concat(0, train_labels)))

  # Optimizer.
  global_step = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(10.0, global_step, 5000, 0.1, staircase=True)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  gradients, v = zip(*optimizer.compute_gradients(loss))
  gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
  optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

  # Predictions.
  train_prediction = tf.nn.softmax(logits) 
  
  # Sampling and validation eval: batch 1, no unrolling.
  sample_input = [tf.placeholder(tf.float32, shape=[1, vocabulary_size]) for _ in range(num_grams)]
#   sample_input = list()
#   for _ in range(num_grams):
#     sample_input.append(tf.placeholder(tf.float32, shape=[1, vocabulary_size]))
  ids = tf.argmax(sample_input[0],1) * vocabulary_size + tf.argmax(sample_input[1],1)
  embedded_sample_input = tf.nn.embedding_lookup(embeddings, ids)
  saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
  saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
  reset_sample_state = tf.group(
    saved_sample_output.assign(tf.zeros([1, num_nodes])),
    saved_sample_state.assign(tf.zeros([1, num_nodes])))
  sample_output, sample_state = lstm_cell(
    embedded_sample_input, saved_sample_output, saved_sample_state)
  with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                saved_sample_state.assign(sample_state)]):
    sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))

# COMMAND ----------

num_steps = 7001
summary_frequency = 100

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  mean_loss = 0
  for step in range(num_steps):
    batches = train_batches.next()
    feed_dict = dict()
    for i in range(num_unrollings + 1):
      feed_dict[train_data[i]] = batches[i]
      
    _, l, predictions, lr = session.run([optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
    mean_loss += l
    if step % summary_frequency == 0:
      if step > 0:
        mean_loss = mean_loss / summary_frequency
      # The mean loss is an estimate of the loss over the last few batches.
      print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
      mean_loss = 0
      labels = np.concatenate(list(batches)[2:]) 
      print('Minibatch perplexity: %.2f' % float(np.exp(logprob(predictions, labels))))

      if step % (summary_frequency * 10) == 0:
        # Generate some samples.
        print('=' * 80)
        for _ in range(5):
          feed = collections.deque(maxlen=num_grams)
          for _ in range(num_grams):
            feed.append(random_distribution())
          sentence = characters(feed[0])[0] + characters(feed[1])[0]
          reset_sample_state.run()
          for _ in range(79):
            prediction = sample_prediction.eval({sample_input[0]: feed[0],
                                                sample_input[1]: feed[1]})
            feed.append(sample(prediction))
            sentence += characters(feed[1])[0]
          print(sentence)
        print('=' * 80)
  
      # Measure validation set perplexity.
      reset_sample_state.run()
      valid_logprob = 0
      for _ in range(valid_size):
        b = valid_batches.next()
        predictions = sample_prediction.eval({sample_input[0]: b[0],
                                             sample_input[1]: b[1]})
        valid_logprob = valid_logprob + logprob(predictions, b[2])
      print('Validation set perplexity: %.2f' % float(np.exp(valid_logprob / valid_size)))

# COMMAND ----------

num_steps = 7001
summary_frequency = 100

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  mean_loss = 0
  for step in range(num_steps):
    batches = train_batches.next()
    feed_dict = dict()
    for i in range(num_unrollings + 1):
      feed_dict[train_data[i]] = batches[i]
      
    _, l, predictions, lr = session.run([optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
    mean_loss += l
    if step % summary_frequency == 0:
      if step > 0:
        mean_loss = mean_loss / summary_frequency
      # The mean loss is an estimate of the loss over the last few batches.
      print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
      mean_loss = 0
      labels = np.concatenate(list(batches)[2:]) 
      print('Minibatch perplexity: %.2f' % float(np.exp(logprob(predictions, labels))))

      if step % (summary_frequency * 10) == 0:
        # Generate some samples.
        print('=' * 80)
        for _ in range(5):
          feed = collections.deque(maxlen=num_grams)
          for _ in range(num_grams):
            feed.append(random_distribution())
          sentence = characters(feed[0])[0] + characters(feed[1])[0]
          reset_sample_state.run()
          for _ in range(79):
            prediction = sample_prediction.eval({sample_input[i]: feed[i] for i in range(num_grams)})
            feed.append(sample(prediction))
            sentence += characters(feed[1])[0]
          print(sentence)
        print('=' * 80)
  
      # Measure validation set perplexity.
      reset_sample_state.run()
      valid_logprob = 0
      for _ in range(valid_size):
        b = valid_batches.next()
        prediction = sample_prediction.eval({sample_input[i]: b[i] for i in range(num_grams)})
        valid_logprob = valid_logprob + logprob(prediction, b[2])
      print('Validation set perplexity: %.2f' % float(np.exp(valid_logprob / valid_size)))

# COMMAND ----------

num_nodes = 64
embedding_size = 128
num_grams = 2
keep_prob = 0.75

# 0.5=8.90, dropout between cells and in the connected layer
# 0.8=6.86, dropout between cells and in the connected layer
# 0.5=6.76, no dropout in the connected layer

graph = tf.Graph()
with graph.as_default():
  
  # Parameters:
  # Input/Forget/Update/Output gate: input, previous output, and bias.  
  ifcox = tf.Variable(tf.truncated_normal([embedding_size, 4 * num_nodes], -0.1, 0.1))
  ifcom = tf.Variable(tf.truncated_normal([num_nodes, 4 * num_nodes], -0.1, 0.1))
  ifcob = tf.Variable(tf.zeros([1, 4 * num_nodes]))
  embeddings = tf.Variable(tf.random_uniform([vocabulary_size * vocabulary_size, embedding_size], -1.0, 1.0))
  
  # Variables saving state across unrollings.
  saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  # Classifier weights and biases.
  w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
  b = tf.Variable(tf.zeros([vocabulary_size]))
  
  # Definition of the cell computation.
  def lstm_cell(i, o, state):
    all_gates = tf.matmul(i, ifcox) + tf.matmul(o, ifcom) + ifcob
    input_gate = tf.sigmoid(all_gates[:, :num_nodes])
    forget_gate = tf.sigmoid(all_gates[:, num_nodes:2*num_nodes])
    update = tf.sigmoid(all_gates[:, 2*num_nodes:3*num_nodes])
    state = forget_gate * state + input_gate * tf.tanh(update)
    output_gate = tf.sigmoid(all_gates[:, 3*num_nodes:])
    return output_gate * tf.tanh(state), state

  # Input data.
  train_data = list()
  for _ in range(num_unrollings + 1):
    train_data.append(tf.placeholder(tf.float32, shape=[batch_size,vocabulary_size]))
  train_inputs = zip(train_data[:-2:], train_data[1:-1:])
  train_labels = train_data[2:]
  
  # Unrolled LSTM loop.
  outputs = list()
  output = saved_output
  state = saved_state
  for i in train_inputs:
    ids = tf.argmax(i[0],1) * vocabulary_size + tf.argmax(i[1],1)
    embedded_i = tf.nn.embedding_lookup(embeddings, ids)
    dropout_i = tf.nn.dropout(embedded_i, keep_prob)
    output, state = lstm_cell(dropout_i, output, state)
    outputs.append(output)
  
  # State saving across unrollings.
  with tf.control_dependencies([saved_output.assign(output),
                                saved_state.assign(state)]):
    # Classifier.
    logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
    # dropout_logits = tf.nn.dropout(logits, keep_prob = keep_prob)
    loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits, tf.concat(0, train_labels)))

  # Optimizer.
  global_step = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(10.0, global_step, 5000, 0.1, staircase=True)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  gradients, v = zip(*optimizer.compute_gradients(loss))
  gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
  optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

  # Predictions.
  train_prediction = tf.nn.softmax(logits) 
  
  # Sampling and validation eval: batch 1, no unrolling.
  sample_input = [tf.placeholder(tf.float32, shape=[1, vocabulary_size]) for _ in range(num_grams)]
  ids = tf.argmax(sample_input[0],1) * vocabulary_size + tf.argmax(sample_input[1],1)
  embedded_sample_input = tf.nn.embedding_lookup(embeddings, ids)
  saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
  saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
  reset_sample_state = tf.group(
    saved_sample_output.assign(tf.zeros([1, num_nodes])),
    saved_sample_state.assign(tf.zeros([1, num_nodes])))
  sample_output, sample_state = lstm_cell(
    embedded_sample_input, saved_sample_output, saved_sample_state)
  with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                saved_sample_state.assign(sample_state)]):
    sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))

# COMMAND ----------

num_steps = 7001
summary_frequency = 100

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  mean_loss = 0
  for step in range(num_steps):
    batches = train_batches.next()
    feed_dict = dict()
    for i in range(num_unrollings + 1):
      feed_dict[train_data[i]] = batches[i]
      
    _, l, predictions, lr = session.run([optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
    mean_loss += l
    if step % summary_frequency == 0:
      if step > 0:
        mean_loss = mean_loss / summary_frequency
      # The mean loss is an estimate of the loss over the last few batches.
      print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
      mean_loss = 0
      labels = np.concatenate(list(batches)[2:]) 
      print('Minibatch perplexity: %.2f' % float(np.exp(logprob(predictions, labels))))

      if step % (summary_frequency * 10) == 0:
        # Generate some samples.
        print('=' * 80)
        for _ in range(5):
          feed = collections.deque(maxlen=num_grams)
          for _ in range(num_grams):
            feed.append(random_distribution())
          sentence = characters(feed[0])[0] + characters(feed[1])[0]
          reset_sample_state.run()
          for _ in range(79):
            prediction = sample_prediction.eval({sample_input[i]: feed[i] for i in range(num_grams)})
            feed.append(sample(prediction))
            sentence += characters(feed[1])[0]
          print(sentence)
        print('=' * 80)
  
      # Measure validation set perplexity.
      reset_sample_state.run()
      valid_logprob = 0
      for _ in range(valid_size):
        b = valid_batches.next()
        prediction = sample_prediction.eval({sample_input[i]: b[i] for i in range(num_grams)})
        valid_logprob = valid_logprob + logprob(prediction, b[2])
      print('Validation set perplexity: %.2f' % float(np.exp(valid_logprob / valid_size)))

# COMMAND ----------

# MAGIC %md
# MAGIC Write a sequence-to-sequence LSTM which mirrors all the words in a sentence. For example, if your input is:
# MAGIC 
# MAGIC `the quick brown fox`
# MAGIC 
# MAGIC the model should attempt to output:
# MAGIC 
# MAGIC `eht kciuq nworb xof`
# MAGIC 
# MAGIC Refer to the lecture on how to put together a sequence-to-sequence model, as well as [this article](http://arxiv.org/abs/1409.3215) for best practices.

# COMMAND ----------

