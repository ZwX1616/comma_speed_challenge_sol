import tensorflow as tf

import csv
import numpy as np
from scipy import misc

# ending index of training set
train_end = 19399

# when get_batch(), returns the next batch for training
# when all batches finished, call it a epoch and re-shuffle
class DataLoader_train:
	def __init__(self, batch_size):
		self.batch_size = batch_size
		self.data_shape = (64,192,1) # data_shape = (H,W,C)

		# training set indexes
		# self.train_index = np.array([i+1 for i in range(train_end)]) #trivial

		# load the ground truths
		self.label = []
		with open('./data/train.txt',encoding='utf-8') as cfile:
			reader = csv.reader(cfile)
			readeritem=[]
			readeritem.extend([row for row in reader])
		for i, row in enumerate(readeritem):
			self.label.append(float(row[0]))
			if i==train_end:
				break
		del reader
		del readeritem
		self.label = np.array(self.label)

		# initialize batches
		self.reset_epoch()

	def load_and_preprocess(self, image_id):
		img = misc.imread('./data/image_train/'+str(image_id)+'.jpg')
		img = img / 255.0;
		return img.reshape((1,self.data_shape[0],self.data_shape[1],self.data_shape[2]))

	def reset_epoch(self):
		self.train_index = np.random.permutation(train_end) + 1
		self.current_batch = 0
		self.is_complete = False
		print('batches rescrambled')

	def get_batch(self):
	# returns np.array x,y
	# x with shape (batch, H, W, 2)
	# y with shape (batch, 1)
		if (self.is_complete==True):
			print ("have finished reading all batches. please reset_epoch().")
			return []

		img = []
		if (len(self.train_index)-self.batch_size*(self.current_batch+1)>0):
		# remaining images more than batch_size
			for i in range(self.batch_size):
				current_frame = self.load_and_preprocess(self.train_index[self.batch_size*self.current_batch+i])
				# img.append(current_frame)
				last_frame = self.load_and_preprocess(self.train_index[self.batch_size*self.current_batch+i]-1)
				img.append(np.concatenate((current_frame,last_frame),axis=-1))
			lbl = self.label[self.train_index[self.batch_size*self.current_batch:self.batch_size*(self.current_batch+1)]]
			self.current_batch = self.current_batch + 1
		else:
		# remaining images less than or equal batch_size (last batch)
			for i in range(len(self.train_index)-self.batch_size*self.current_batch):
				current_frame = self.load_and_preprocess(self.train_index[self.batch_size*self.current_batch+i])
				# img.append(current_frame)
				last_frame = self.load_and_preprocess(self.train_index[self.batch_size*self.current_batch+i]-1)
				img.append(np.concatenate((current_frame,last_frame),axis=-1))
			lbl = self.label[self.train_index[self.batch_size*self.current_batch:]]
			self.current_batch = self.current_batch + 1
			self.is_complete=True

		return np.concatenate(img,axis=0), \
				np.reshape(lbl,(len(img),1))