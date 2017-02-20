from __future__ import absolute_import
import numpy as np
import tensorflow as tf

# convolution/pool stride
_CONV_KERNEL_STRIDES_ = [1, 2, 2, 1]
_DECONV_KERNEL_STRIDES_ = [1, 2, 2, 1]
_REGULAR_FACTOR_ = 1.0e-4

def _construct_conv_layer(input_layer, output_dim, kernel_size = 3, stddev = 0.02, name = 'conv2d'):
	with tf.variable_scope(name):
		init_weight = tf.truncated_normal_initializer(mean = 0.0, stddev = stddev, dtype = tf.float32)
		filter_size = [kernel_size, kernel_size, input_layer.get_shape()[-1], output_dim]
		weight = tf.get_variable(
			name = name + 'weight',
			shape = filter_size,
			initializer = init_weight,
			regularizer = tf.contrib.layers.l2_regularizer(_REGULAR_FACTOR_))
		bias = tf.get_variable(
			name = name + 'bias',
			shape = [output_dim],
			initializer = tf.constant_initializer(0.0))
		conv = tf.nn.conv2d(input_layer, weight, _CONV_KERNEL_STRIDES_, padding = 'SAME')
		conv = tf.nn.bias_add(conv, bias)
		return conv

def _construct_deconv_layer(input_layer, output_shape, kernel_size = 2, stddev = 0.02, name = 'deconv'):
	with tf.variable_scope(name):
		init_weight = tf.truncated_normal_initializer(mean = 0.0, stddev = stddev, dtype = tf.float32)
		filter_size = [kernel_size, kernel_size, output_shape[-1], input_layer.get_shape()[-1]]
		weight = tf.get_variable(
			name = name + 'weight',
			shape = filter_size,
			initializer = init_weight,
			regularizer = tf.contrib.layers.l2_regularizer(_REGULAR_FACTOR_))
		bias = tf.get_variable(
			name = name + 'bias',
			shape = [output_shape[-1]],
			initializer = tf.constant_initializer(0.0))
		deconv = tf.nn.conv2d_transpose(input_layer, weight, output_shape, strides = _DECONV_KERNEL_STRIDES_, padding = 'SAME')
		deconv = tf.nn.bias_add(deconv, bias)
		return deconv

def _construct_lrelu(input_layer, leak = 0.2, name = 'lrelu'):
	with tf.variable_scope(name):
		alpha1 = 0.5 * (1 + leak)
		alpha2 = 0.5 * (1 - leak)
		return alpha1 * input_layer + alpha2 * abs(input_layer)

def _construct_full_connection_layer(input_layer, output_dim, stddev = 0.02, name = 'fc'):
	# calculate input_layer dimension and reshape to batch * dimension
	input_dimension = 1
	for dim in input_layer.get_shape().as_list()[1:]:
		input_dimension *= dim

	with tf.variable_scope(name):
		init_weight = tf.truncated_normal_initializer(mean = 0.0, stddev = stddev, dtype = tf.float32)
		filter_size = [input_dimension, output_dim]
		weight = tf.get_variable(
			name = name + 'weight',
			shape = filter_size,
			initializer = init_weight,
			regularizer = tf.contrib.layers.l2_regularizer(_REGULAR_FACTOR_))
		bias = tf.get_variable(
			name = name + 'bias',
			shape = [output_dim],
			initializer = tf.constant_initializer(0.0))
		input_layer_reshape = tf.reshape(input_layer, [-1, input_dimension])
		fc = tf.matmul(input_layer_reshape, weight)
		tc = tf.nn.bias_add(fc, bias)
		return fc

class _BatchNormalization:
	def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
		with tf.variable_scope(name):
			self.epsilon  = epsilon
			self.momentum = momentum
			self.name = name

	def __call__(self, x, train=True):
		return tf.contrib.layers.batch_norm(x,
				decay=self.momentum, 
				updates_collections=None,
				epsilon=self.epsilon,
				scale=True,
				is_training=train,
				scope=self.name)

# define DCGAN network
# define disctriminative network
class Discriminative:
	def __init__(self, name = 'discriminator'):
		self.name = name
		self.d_bn1 = _BatchNormalization(name = 'd_bn1')

	def inference(self, images, n_class = 10, cont_dim = 2,reuse = False):
		with tf.variable_scope(self.name) as scope:
			if reuse:
				scope.reuse_variables()
			print 'discriminator input image:', images.get_shape()

			hidden0 = _construct_lrelu(_construct_conv_layer(images, 64, name = 'd_conv_hidden0'))
			hidden1 = _construct_lrelu(self.d_bn1(_construct_conv_layer(hidden0, 128, name = 'd_conv_hidden1')))
			fc0 = tf.nn.relu(_construct_full_connection_layer(hidden1, 1024, name = 'd_fc_hidden2'))
			fc1 = tf.nn.relu(_construct_full_connection_layer(fc0, 128, name = 'd_fc_hidden3'))
			output_dis = _construct_full_connection_layer(fc1, 1, name = 'd_out_dis')
			output_cat = _construct_full_connection_layer(fc1, n_class, name = 'd_out_cat')
			output_cont = _construct_full_connection_layer(fc1, cont_dim, name = 'd_out_cont')
			print 'discriminator ouput dis:', output_dis.get_shape()
			print 'discriminator ouput cat:', output_dis.get_shape()
			print 'discriminator ouput cont:', output_dis.get_shape()

			return output_dis, output_cat, output_cont

class Generative:
	def __init__(self, name = 'generator'):
		self.name = name
		self.g_bn0 = _BatchNormalization(name = 'g_bn0')
		self.g_bn1 = _BatchNormalization(name = 'g_bn1')
		self.g_bn2 = _BatchNormalization(name = 'g_bn2')

	def inference(self, z, image_shape = [28, 28, 1], reuse = False):
		with tf.variable_scope(self.name) as scope:
			if reuse:
				scope.reuse_variables()

			print 'generator input z:', z.get_shape()
			batch_size = 64
			fc = _construct_full_connection_layer(z, 1024, name = 'g_fc_hidden0')
			fc = tf.nn.relu(self.g_bn0(fc))
			fc = _construct_full_connection_layer(z, 7 * 7 * 128, name = 'g_fc_hidden1')
			fc_reshape = tf.reshape(fc, [-1, 7, 7, 128])
			fc_reshape = tf.nn.relu(self.g_bn1(fc_reshape))
			batch_size = fc_reshape.get_shape().as_list()[0]
			deconv0 = _construct_deconv_layer(fc_reshape, [batch_size, 14, 14, 64], name = 'g_deconv_hidden1')
			deconv0 = tf.nn.relu(self.g_bn2(deconv0))
			deconv1 = _construct_deconv_layer(deconv0, [batch_size, image_shape[0], image_shape[1], image_shape[2]], name = 'g_deconv_hidden2')
			output_layer = tf.nn.tanh(deconv1)
			print "generator output:", output_layer.get_shape()

			return output_layer

class InfoGAN:
	def __init__(self, discriminator_name = 'discriminator', generator_name = 'generator'):
		self.discriminator_name = discriminator_name
		self.generator_name = generator_name

	def inference(self, images, z, n_class = 10, cont = 2):
		# generative
		self.image_shape = images.get_shape().as_list()[1:]
		self.generator = Generative(self.generator_name)
		self.g_images = self.generator.inference(z, self.image_shape)

		# discriminative
		self.discriminator = Discriminative(self.discriminator_name)
		self.real_dis, _, _ = self.discriminator.inference(images, n_class, cont)
		self.z_dis, self.z_cat, self.z_con = self.discriminator.inference(self.g_images, n_class, cont, reuse = True)

		return self.real_dis, self.z_dis, self.z_cat, self.z_con

	def generate_images(self, z,row=8, col=8):
		images = tf.cast(tf.mul(tf.add(self.generator.inference(z, self.image_shape, reuse = True), 1.0), 127.5), tf.uint8)
		batch_size = images.get_shape().as_list()[0]
		images = [image for image in tf.split(0, batch_size, images)]
		rows = []
		for i in range(row):
			rows.append(tf.concat(2, images[col * i + 0:col * i + col]))
		image = tf.concat(1, rows)
		return tf.image.encode_png(tf.squeeze(image, [0]))
