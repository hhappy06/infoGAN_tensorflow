from __future__ import absolute_import
import gzip, os, six, sys
import numpy as np
from PIL import Image

_MNIST_PATH_ = './data/mnist/'
_FILE_NAMES_ = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']

n_train = 60000
n_test = 10000
dim = 28 * 28

def load_mnist(data_filename, label_filename, num):
	images = np.zeros(num * dim, dtype=np.uint8).reshape((num, dim))
	label = np.zeros(num, dtype=np.uint8).reshape((num, ))
	with gzip.open(data_filename, "rb") as f_images, gzip.open(label_filename, "rb") as f_labels:
		f_images.read(16)
		f_labels.read(8)
		for i in six.moves.range(num):
			label[i] = ord(f_labels.read(1))
			for j in six.moves.range(dim):
				images[i, j] = ord(f_images.read(1))

			if i % 100 == 0 or i == num - 1:
				sys.stdout.write("\rloading images ... ({} / {})".format(i + 1, num))
				sys.stdout.flush()
	sys.stdout.write("\n")
	return images, label

def extract_bitmaps():
	train_dir = "./data/mnist/train_images"
	test_dir = "./data/mnist/test_images"
	try:
		os.mkdir(train_dir)
		os.mkdir(test_dir)
	except:
		pass
	data_train, label_train = load_mnist(_MNIST_PATH_ + '/' + _FILE_NAMES_[0], _MNIST_PATH_ + '/' + _FILE_NAMES_[1], n_train)
	data_test, label_test = load_mnist(_MNIST_PATH_ + '/' + _FILE_NAMES_[2], _MNIST_PATH_ + '/' + _FILE_NAMES_[3], n_test)
	o_tain_file_list = open(train_dir + '/file_list', 'w')
	o_test_file_list = open(test_dir + '/file_list', 'w')
	print "Saving training images ..."
	for i in xrange(data_train.shape[0]):
		image = Image.fromarray(data_train[i].reshape(28, 28))
		image.save("{}/{}_{}.png".format(train_dir, label_train[i], i))
		o_tain_file_list.write('{}_{}.png,{}\n'.format(label_train[i], i, label_train[i]))
	print "Saving test images ..."
	for i in xrange(data_test.shape[0]):
		image = Image.fromarray(data_test[i].reshape(28, 28))
		image.save("{}/{}_{}.png".format(test_dir, label_test[i], i))
		o_test_file_list.write('{}_{}.png,{}\n'.format(label_test[i], i, label_test[i]))
	o_tain_file_list.close()
	o_test_file_list.close()

if __name__ == '__main__':
	extract_bitmaps()