from __future__ import absolute_import

import numpy as np
import tensorflow as tf

_IMAGE_HEIGHT_ORG_ = 28
_IMAGE_WIDTH_ORG_ = 28
_IMAGE_DEPTH_ORG_ = 1

_IMAGE_SIZE_ = 28

class ImageParser():
	def parse(self, value, data_dir = ''):
		file_name, label = tf.decode_csv(value, [[''], [0]])
		if data_dir:
			file_name = data_dir + '/' + file_name
		png = tf.read_file(file_name)
		print file_name
		image = tf.image.decode_png(png, channels = 1)
		# normalize image to [-1, 1]
		image = tf.cast(image, tf.float32) * (2.0/255.0) - 1
		image.set_shape([_IMAGE_HEIGHT_ORG_, _IMAGE_WIDTH_ORG_, _IMAGE_DEPTH_ORG_])
		resize_image = tf.image.resize_images(image, [_IMAGE_SIZE_, _IMAGE_SIZE_])
		# image precess if necessary
		label = tf.cast(label, tf.int64)

		return resize_image, label
