import os, sys
import tensorflow as tf
import numpy as np
import time
from model.infogan import InfoGAN
from input.line_parser.line_parser import ImageParser
from input.data_reader import read_data
from loss.infogan_loss import InfoGANLoss
from train_op import d_train_opt, g_train_opt

_Z_DIM_ = 10
_Z_CONT_DIM = 2
_N_CLASS_ = 10

_BATCH_SIZE_ = 64
_EPOCH_ = 10
_TRAINING_SET_SIZE_ = 60000
_DATA_DIR_ = './data/mnist/train_images'
_CSVFILE_ = ['./data/mnist/train_images/file_list']

_OUTPUT_INFO_FREQUENCE_ = 100
_OUTPUT_IMAGE_FREQUENCE_ = 100

line_parser = ImageParser()
infogan_loss = InfoGANLoss()

def train():
	with tf.Graph().as_default():
		# input
		images, labels = read_data(_CSVFILE_, line_parser = line_parser, data_dir = _DATA_DIR_, batch_size = _BATCH_SIZE_)

		z_cat = tf.multinomial(tf.ones((_BATCH_SIZE_, _N_CLASS_), dtype=tf.float32) / _N_CLASS_, 1)
		z_cat = tf.squeeze(z_cat, -1)
		# continuous latent variable
		z_con = tf.random_normal((_BATCH_SIZE_, _Z_CONT_DIM))
		z_rand = tf.random_normal((_BATCH_SIZE_, _Z_DIM_))
		z = tf.concat(1, [tf.one_hot(z_cat, depth = _N_CLASS_), z_con, z_rand])

		# model 
		infogan = InfoGAN('discriminator', 'generator')
		out_real_dis, out_z_dis, out_z_cat, out_z_con = infogan.inference(images, z, n_class = _N_CLASS_, cont = _Z_CONT_DIM)
		d_loss, g_loss = infogan_loss.loss(out_real_dis, out_z_dis, out_z_cat, out_z_con, z_cat, z_con)

		# summary
		sum_z = tf.summary.histogram('z', z)
		sum_d_loss = tf.summary.scalar('d_loss', d_loss)
		sum_g_loss = tf.summary.scalar('g_loss', g_loss)

		sum_g = tf.summary.merge([sum_z, sum_g_loss])
		sum_d = tf.summary.merge([sum_z, sum_d_loss])

		# opt
		trainable_vars = tf.trainable_variables()
		d_vars = [var for var in trainable_vars if 'discriminator' in var.name]
		g_vars = [var for var in trainable_vars if 'generator' in var.name]

		d_opt = d_train_opt(d_loss, d_vars)
		g_opt = g_train_opt(g_loss, g_vars)

		# generate_images for showing
		generate_images = infogan.generate_images(z, 4, 4)
		
		# initialize variable
		init_op = tf.global_variables_initializer()
		saver = tf.train.Saver(tf.global_variables())

		session = tf.Session()
		file_writer = tf.summary.FileWriter('./logs', session.graph)
		session.run(init_op)

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=session, coord=coord)

		print 'InfoGAN training starts...'
		sys.stdout.flush()
		counter = 0
		max_steps = int(_TRAINING_SET_SIZE_ / _BATCH_SIZE_)
		for epoch in xrange(_EPOCH_):
			for step in xrange(max_steps):

				_, summary_str, error_d_loss = session.run([d_opt, sum_d, d_loss])
				file_writer.add_summary(summary_str, counter)

				_, summary_str, error_g_loss = session.run([g_opt, sum_g, g_loss])
				file_writer.add_summary(summary_str, counter)

				_, summary_str, error_g_loss = session.run([g_opt, sum_g, g_loss])
				file_writer.add_summary(summary_str, counter)

				file_writer.flush()

				counter += 1

				if counter % _OUTPUT_INFO_FREQUENCE_ == 0:
					print 'step: (%d, %d), d_loss: %f, g_loss:%f'%(epoch, step, error_d_loss, error_g_loss)
					sys.stdout.flush()

				if counter % _OUTPUT_IMAGE_FREQUENCE_ == 0:
					generated_image_eval = session.run(generate_images)
					filename = os.path.join('./result', 'out_%03d_%05d.png' %(epoch, step))
					with open(filename, 'wb') as f:
						f.write(generated_image_eval)
					print 'output generated image: %s'%(filename)
					sys.stdout.flush()

		print 'training done!'
		file_writer.close()
		coord.request_stop()
		coord.join(threads)
		session.close()

if __name__ == '__main__':
	train()
