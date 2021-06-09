'''
Sanjay Singh
san.singhsanjay@gmail.com
June-2021
To read the mnist data in the "dataset" directory of this repo
'''

# packages
import idx2numpy
import numpy as np

# data paths
train_images_path = "/home/sansingh/github_repo/mxnet_mnist_cpp_simpleNN/dataset/mnist_data/train-images.idx3-ubyte"
train_labels_path = "/home/sansingh/github_repo/mxnet_mnist_cpp_simpleNN/dataset/mnist_data/train-labels.idx1-ubyte"
val_images_path = "/home/sansingh/github_repo/mxnet_mnist_cpp_simpleNN/dataset/mnist_data/t10k-images.idx3-ubyte"
val_labels_path = "/home/sansingh/github_repo/mxnet_mnist_cpp_simpleNN/dataset/mnist_data/t10k-labels.idx1-ubyte"

# reading above dataset
train_images = idx2numpy.convert_from_file(train_images_path)
train_labels = idx2numpy.convert_from_file(train_labels_path)
val_images = idx2numpy.convert_from_file(val_images_path)
val_labels = idx2numpy.convert_from_file(val_labels_path)

# print status of data
print("Train Data: ", train_images.shape)
print("Train Labels: ", train_labels.shape)
print("Val Data: ", val_images.shape)
print("Val Labels: ", val_labels.shape)
