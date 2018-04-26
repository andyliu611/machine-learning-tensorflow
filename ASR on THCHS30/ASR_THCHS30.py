# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ ASR_THCHS30.py ]
#   Synopsis     [ automatic speech recognition on the THCHS30 dataset - tensorflow ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
#   Reference    [ http://blog.topspeedsnail.com/archives/10696 ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import csv
import pickle
import random
import argparse
import numpy as np
import tensorflow as tf
from collections import Counter
import librosa  # >> https://github.com/librosa/librosa


##################
# CONFIGURATIONS #
##################
def get_config():
	parser = argparse.ArgumentParser(description='ASR_THCHS30 configuration')
	#--mode--#
	parser.add_argument('--train', action='store_true', help='run training process')
	parser.add_argument('--reprocess', action='store_true', help='process and read all wav files to obtain max len, this may take a while')
	#--parameters--#
	parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
	parser.add_argument('--n_epoch', type=int, default=100, help='number of training epoch')
	parser.add_argument('--lr', type=float, default=0.001, help='learning rate for optimizer')
	#--path--#
	parser.add_argument('--data_path', type=str, default='./data_thchs30/data/', help='path to the THCHS30 file')
	parser.add_argument('--model_dir', type=str, default='./model', help='model storage directory')
	config = parser.parse_args()
	return config

 
#################
# GET WAV FILES #
#################
def get_wav_files(path):
	wav_files = []
	for (dirpath, dirnames, filenames) in os.walk(path):
		for filename in filenames:
			if filename.endswith('.wav'):
				filename_path = os.sep.join([dirpath, filename])
				if os.stat(filename_path).st_size < 240000:  # ignore small files
					continue
				wav_files.append(filename_path)
	return wav_files
 

#################
# GET WAV LABEL #
#################
def get_wav_lable(path, wav_files):
	labels = []
	label_ids = []
	#--get id list--#
	for wav_file in wav_files:
		wav_id = os.path.basename(wav_file).split('.')[0]
		label_ids.append(wav_id)
	#--get labels that are in id list--#
	for (dirpath, dirnames, filenames) in os.walk(path):
		for filename in filenames:
			if filename.endswith('.trn'):
				filename_path = os.sep.join([dirpath, filename])
				label_id = os.path.basename(filename_path).split('.')[0]
				if label_id in label_ids:
					with open(filename_path, 'r') as f:
						label_line = f.readline()
						if label_line[-1] == '\n': label_line = label_line[:-1]
						labels.append(label_line)
	assert len(wav_files) == len(labels)
	print('Number of training samples: ', len(wav_files)) # >> 11841
	return labels
	

#################
# BUILD MAPPING #
#################
def build_mapping(labels):
	all_words = []
	for label in labels:
		all_words += [word for word in label]
	counter = Counter(all_words)
	count_pairs = sorted(counter.items(), key=lambda x: -x[1])
	 
	words, _ = zip(*count_pairs)
	print('>> building vocabulary, vocab size:', len(words)) # >> 2882
	 
	word2idx = dict(zip(words, range(len(words))))
	word2idx['<unk>'] = len(words)
	idx2word = { i : w for w, i in word2idx.items() }
	return word2idx, idx2word


####################
# MAP LABEL TO IDX #
####################
def label2idx(word2idx, labels):
	to_num = lambda word: word2idx.get(word, len(word2idx))
	labels_in_idx = [ list(map(to_num, label)) for label in labels]
	# >> [466, 0, 9, 0, 158, 280, 0, 231, 0, 599, 0, 23, 332, 0, 25, 1200, 0, 1, 0, 516, 218, 0, 65, 40, 0, 1, 0, 312, 0, 1323, 0, 272, 9, 0, 466, 0, 70, 0, 810, 274, 0, 748, 1833, 0, 1067, 154, 0, 2111, 85]
	label_max_len = np.max([len(label) for label in labels_in_idx])
	print('Max length of label: ', label_max_len) # >> 75
	return labels_in_idx, label_max_len
 

#####################
# GET MAX AUDIO LEN #
#####################
def get_max_audio_len(wav_files):
	wav_max_len = 0  
	for wav in wav_files:
		wav, sr = librosa.load(wav, mono=True)
		mfcc = np.transpose(librosa.feature.mfcc(wav, sr), [1,0])
		if len(mfcc) > wav_max_len:
			wav_max_len = len(mfcc)
	print('Max audio length', wav_max_len) # >> 673
	return wav_max_len


#################
# CLASS ASR NET #
#################
class ASR_NET(object):


	##################
	# INITIALIZATION #
	##################
	def __init__(self, config, vocab_size, label_max_len, wav_max_len, wav_files, labels_in_idx):

		#--parameters--#
		self.batch_size = config.batch_size
		self.n_epoch = config.n_epoch
		self.lr = config.lr
		self.model_dir = config.model_dir

		#--len--#
		self.vocab_size = vocab_size
		self.label_max_len = label_max_len
		self.wav_max_len = wav_max_len

		#--data--#
		self.wav_files = wav_files
		self.labels = labels_in_idx

		#--placeholders--#
		self.X = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, 20])
		self.Y = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, None])

		#--model variables--#
		self.sequence_len = tf.reduce_sum(tf.cast(tf.not_equal(tf.reduce_sum(self.X, axis=2), 0.), tf.int32), axis=1)
		self.n_batch = len(self.wav_files) // self.batch_size
		self.best_loss = 777.777
		self.batch_pointer = 0

		#--build model--#
		self.logit = self.build_network()

		#--model saver--#
		self.variables = [var for var in tf.global_variables() if 'ASR_NET' in var.name]
		self.saver = tf.train.Saver(var_list=self.variables, max_to_keep=1)


	#################
	# BUILD NETWORK #
	#################
	def build_network(self, n_dim=128, n_blocks=3):
		with tf.variable_scope('ASR_NET'):
			out = self.conv1d_layer(layer_id=0, input_tensor=self.X, size=1, dim=n_dim, activation='tanh', scale=0.14, bias=False)
			# skip connections
			def residual_block(input_sensor, size, rate, id_increment):
					conv_filter = self.aconv1d_layer(layer_id=1+id_increment, input_tensor=input_sensor, size=size, rate=rate, activation='tanh', scale=0.03, bias=False)
					conv_gate = self.aconv1d_layer(layer_id=2+id_increment, input_tensor=input_sensor, size=size, rate=rate,  activation='sigmoid', scale=0.03, bias=False)
					out = conv_filter * conv_gate
					out = self.conv1d_layer(layer_id=3+id_increment, input_tensor=out, size=1, dim=n_dim, activation='tanh', scale=0.08, bias=False)
					return out + input_sensor, out
			skip = 0
			id_increment = 0
			for _ in range(n_blocks):
				for r in [1, 2, 4, 8, 16]:
					out, s = residual_block(out, size=7, rate=r, id_increment=id_increment)
					id_increment += 3
					skip += s
		 
			logit = self.conv1d_layer(layer_id=id_increment+1, input_tensor=skip, size=1, dim=skip.get_shape().as_list()[-1], activation='tanh', scale=0.08, bias=False)
			logit = self.conv1d_layer(layer_id=id_increment+2, input_tensor=logit, size=1, dim=self.vocab_size, activation=None, scale=0.04, bias=True)
			return logit
 
 
	#################
	# TRAIN NETWORK #
	#################
	def train(self):		
		#--CTC loss--#
		indices = tf.where(tf.not_equal(tf.cast(self.Y, tf.float32), 0.))
		target = tf.SparseTensor(indices=indices, values=tf.gather_nd(self.Y, indices) - 1, dense_shape=tf.cast(tf.shape(self.Y), tf.int64))
		loss = tf.nn.ctc_loss(labels=target, inputs=self.logit, sequence_length=self.sequence_len, time_major=False)
		
		#--optimizer--#
		optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=0.99)
		var_list = [t for t in tf.trainable_variables()]
		gradient = optimizer.compute_gradients(loss, var_list=var_list)
		optimizer_op = optimizer.apply_gradients(gradient)
		
		#--training session--#
		with tf.Session() as sess:

			#--initialize training--#
			sess.run(tf.global_variables_initializer())
			best_loss = 777.777
			history_loss = []

			#--run epoch--#
			for epoch in range(self.n_epoch):
				
				#--initialize epoch--#
				self.batch_pointer = 0
				epoch_loss = []
				self.shuffle_data()
				
				#--run batch--#
				for batch in range(self.n_batch):
					batches_wavs, batches_labels = self.get_batch()
					train_loss, _ = sess.run([loss, optimizer_op], feed_dict={self.X: batches_wavs, self.Y: batches_labels})
					epoch_loss.append(np.mean(train_loss))
					print('Epoch: %i/%i, Batch: %i/%i, Loss: %.5f' % (epoch, self.n_epoch, batch, self.n_batch, epoch_loss[-1]), end='\r')

				#--epoch checkpoint--#
				history_loss.append(np.mean(epoch_loss))
				to_save = self.model_checkpoint_save_best(cur_val=history_loss[-1], mode='min')
				print('Epoch: %i/%i, Batch: %i/%i, Loss: %.5f, Saved: %s' % (epoch, self.n_epoch, batch, self.n_batch, history_loss[-1], 'True' if to_save else 'False'))
				if to_save == True:
					self.saver.save(sess, os.path.join(self.model_dir, 'ASR_THCHS30'), global_step=(epoch+1))
					pickle.dump(history_loss, open(os.path.join(self.model_dir, 'history_loss.pkl'), 'wb'), True)


	################
	# SHUFFLE DATA #
	################
	def shuffle_data(self):
		to_shuffle = list(zip(self.wav_files, self.labels))
		random.shuffle(to_shuffle)
		self.wav_files, self.labels = zip(*to_shuffle)


	#############
	# GET BATCH #
	#############
	def get_batch(self):
		batches_wavs = []
		batches_labels = []
		for i in range(self.batch_size):
			wav, sr = librosa.load(self.wav_files[self.batch_pointer], mono=True)
			mfcc = np.transpose(librosa.feature.mfcc(wav, sr), [1,0])
			batches_wavs.append(mfcc.tolist())
			batches_labels.append(self.labels[self.batch_pointer])
			self.batch_pointer += 1
	 
		for mfcc in batches_wavs:
			while len(mfcc) < self.wav_max_len:
				mfcc.append([0]*20)
		for label in batches_labels:
			while len(label) < self.label_max_len:
				label.append(0)
		return batches_wavs, batches_labels


	##############
	# _SAVE BEST #
	##############
	"""
		Called by pre_train(), this function checks and saves the best model.
	"""
	def model_checkpoint_save_best(self, cur_val, mode):
		if mode == 'min':
			if cur_val < self.best_loss: 
				self.best_loss = cur_val
				return True
			else: return False
		elif mode == 'max':
			if cur_val > self.best_loss: 
				self.best_loss = cur_val
				return True
			else: return False
		else: raise ValueError('Invalid Mode!')


	################
	# CONV1D LAYER #
	################
	def conv1d_layer(self, layer_id, input_tensor, size, dim, activation, scale, bias):
		global conv1d_index
		with tf.variable_scope('conv1d_' + str(layer_id)):
			W = tf.get_variable('W', (size, input_tensor.get_shape().as_list()[-1], dim), dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-scale, maxval=scale))
			if bias:
				b = tf.get_variable('b', [dim], dtype=tf.float32, initializer=tf.constant_initializer(0))
			out = tf.nn.conv1d(input_tensor, W, stride=1, padding='SAME') + (b if bias else 0)
			if not bias:
				beta = tf.get_variable('beta', dim, dtype=tf.float32, initializer=tf.constant_initializer(0))
				gamma = tf.get_variable('gamma', dim, dtype=tf.float32, initializer=tf.constant_initializer(1))
				mean_running = tf.get_variable('mean', dim, dtype=tf.float32, initializer=tf.constant_initializer(0))
				variance_running = tf.get_variable('variance', dim, dtype=tf.float32, initializer=tf.constant_initializer(1))
				mean, variance = tf.nn.moments(out, axes=list(np.arange(len(out.get_shape()) - 1)))
				def update_running_stat():
					decay = 0.99
					update_op = [mean_running.assign(mean_running * decay + mean * (1 - decay)), variance_running.assign(variance_running * decay + variance * (1 - decay))]
					with tf.control_dependencies(update_op):
						return tf.identity(mean), tf.identity(variance)
					m, v = tf.cond(tf.Variable(False, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES]), update_running_stat, lambda: (mean_running, variance_running))
					out = tf.nn.batch_normalization(out, m, v, beta, gamma, 1e-8)
			if activation == 'tanh':
				out = tf.nn.tanh(out)
			if activation == 'sigmoid':
				out = tf.nn.sigmoid(out)
			return out


	#################
	# ACONV1D LAYER #
	#################
	def aconv1d_layer(self, layer_id, input_tensor, size, rate, activation, scale, bias):
		with tf.variable_scope('aconv1d_' + str(layer_id)):
			shape = input_tensor.get_shape().as_list()
			W = tf.get_variable('W', (1, size, shape[-1], shape[-1]), dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-scale, maxval=scale))
			if bias:
				b = tf.get_variable('b', [shape[-1]], dtype=tf.float32, initializer=tf.constant_initializer(0))
			out = tf.nn.atrous_conv2d(tf.expand_dims(input_tensor, dim=1), W, rate=rate, padding='SAME')
			out = tf.squeeze(out, [1])
			if not bias:
				beta = tf.get_variable('beta', shape[-1], dtype=tf.float32, initializer=tf.constant_initializer(0))
				gamma = tf.get_variable('gamma', shape[-1], dtype=tf.float32, initializer=tf.constant_initializer(1))
				mean_running = tf.get_variable('mean', shape[-1], dtype=tf.float32, initializer=tf.constant_initializer(0))
				variance_running = tf.get_variable('variance', shape[-1], dtype=tf.float32, initializer=tf.constant_initializer(1))
				mean, variance = tf.nn.moments(out, axes=list(np.arange(len(out.get_shape()) - 1)))
				def update_running_stat():
					decay = 0.99
					update_op = [mean_running.assign(mean_running * decay + mean * (1 - decay)), variance_running.assign(variance_running * decay + variance * (1 - decay))]
					with tf.control_dependencies(update_op):
						return tf.identity(mean), tf.identity(variance)
					m, v = tf.cond(tf.Variable(False, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES]), update_running_stat, lambda: (mean_running, variance_running))
					out = tf.nn.batch_normalization(out, m, v, beta, gamma, 1e-8)
			if activation == 'tanh':
				out = tf.nn.tanh(out)
			if activation == 'sigmoid':
				out = tf.nn.sigmoid(out)
			return out
	

"""
def speech_to_text(wav_file):
	wav, sr = librosa.load(wav_file, mono=True)
	mfcc = np.transpose(np.expand_dims(librosa.feature.mfcc(wav, sr), axis=0), [0,2,1])
 
	logit = speech_to_text_network()
 
	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess, tf.train.latest_checkpoint('.'))
 
		decoded = tf.transpose(logit, perm=[1, 0, 2])
		decoded, _ = tf.nn.ctc_beam_search_decoder(decoded, sequence_len, merge_repeated=False)
		predict = tf.sparse_to_dense(decoded[0].indices, decoded[0].shape, decoded[0].values) + 1
		output = sess.run(decoded, feed_dict={X: mfcc})        
		#print(output)
"""

########
# MAIN #
########
"""
	main function
"""
def main():
	config = get_config()

	if config.train:
		wav_files = get_wav_files(path=config.data_path)
		labels = get_wav_lable(path=config.data_path, wav_files=wav_files)

		word2idx, idx2word = build_mapping(labels)
		labels_in_idx, label_max_len = label2idx(word2idx, labels)
		
		vocab_size = len(word2idx)
		if config.reprocess:
			wav_max_len = get_max_audio_len(wav_files)
		else:
			wav_max_len = 703

		ASR = ASR_NET(config, vocab_size, label_max_len, wav_max_len, wav_files, labels_in_idx)
		ASR.train()


if __name__ == '__main__':
	main()

