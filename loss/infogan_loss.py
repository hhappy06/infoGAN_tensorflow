from __future__ import absolute_import
import tensorflow as tf

class InfoGANLoss:
	def loss(self, out_real_dis, out_z_dis, out_z_cat, out_z_cont, z_cat, z_cont):
		d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(out_real_dis, tf.ones_like(out_real_dis)))
		d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(out_z_dis, tf.zeros_like(out_z_dis)))
		g_loss_dis = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(out_z_dis, tf.ones_like(out_z_dis)))
		g_loss_cat = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(out_z_cat, z_cat))
		g_loss_cont = tf.reduce_mean(tf.squared_difference(out_z_cont, z_cont))

		d_loss = d_real_loss + d_fake_loss + g_loss_cat + g_loss_cont
		g_loss = g_loss_dis + g_loss_cat + g_loss_cont

		return d_loss, g_loss