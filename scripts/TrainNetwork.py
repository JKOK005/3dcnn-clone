from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger 
from keras.utils import np_utils

import os
import datetime
import pandas as pd
import numpy as np
from settings import params
from UtilitiesScript import *
from NetController import NetController

K.set_image_dim_ordering("tf")

class TrainNetwork(object):
	def __init__(self, weights_path=None, dropout=None):
		self.net 			= NetController.get_net(load_weight_path=weights_path, USE_DROPOUT=dropout)
		self.train_data 	= None 
		self.train_labels 	= None
		self.val_data 		= None
		self.val_labels 	= None
		self.is_dropout 	= dropout

	def __getCheckPointOptions(self, dir_path):
		file_path 	= os.path.join(dir_path, params["sc_train"]["checkpoint_opt"])
		checkpoint 	= ModelCheckpoint(filepath=file_path, monitor=['val_loss','loss'], verbose=False, save_best_only=False, save_weights_only=False, mode='auto', period=1)
		return checkpoint

	def __is_loaded(self):
		return self.train_data != None and not self.train_labels.empty and self.val_data != None and not self.val_labels.empty

	def __load(self, data_path, labels_path, data_field, labels_field):
		data 		= H5Utilities.read(data_path, data_field)
		labels 		= CsvUtilities.read(labels_path, labels_field)
		data_np 	= data[:]
		return [data_np, labels]

	def loadData(self, train_data_path, train_labels_path, val_data_path, val_labels_path, data_field, labels_field):
		[self.train_data, self.train_labels] 	= self.__load(train_data_path, train_labels_path, data_field, labels_field)
		[self.val_data, self.val_labels] 		= self.__load(val_data_path, val_labels_path, data_field, labels_field)
		return 

	def train(self, batch_size):
		assert self.__is_loaded()

		# Create model storing directory 
		folder_name 	= datetime.datetime.now().strftime("%B_%d_%Y_%H_%M_%S")
		dir_path 		= DirectoryManager.create(params["sc_train"]["model_main_dir"], folder_name, force=False)
		log_path 		= os.path.join(dir_path, params["sc_train"]["log_name"])

		NetLogger.logTraining(log_path, len(self.train_data), sum(self.train_labels), \
					len(self.val_data), sum(self.val_labels), dropout=self.is_dropout)
		checkpoint 		= self.__getCheckPointOptions(dir_path)
		csvlogger 		= CSVLogger(log_path, append=False, separator=',')

		x_train 		= self.train_data
		y_train_class 	= np_utils.to_categorical(self.train_labels, 2)
		x_val 			= self.val_data
		y_val_class 	= np_utils.to_categorical(self.val_labels, 2)

		import IPython
		IPython.embed()
		self.net.fit(x=x_train, y=y_train_class, batch_size=batch_size, shuffle=True, nb_epoch=100, verbose=True, callbacks=[checkpoint,csvlogger], validation_data=(x_val, y_val_class))

if __name__ == "__main__":
	# load model into object
	train_net 		= TrainNetwork(dropout=True)
	sc_split 		= params["sc_split"]

	# Start training
	train_net.loadData(sc_split["train_data"], sc_split["train_label"], sc_split["test_data"], sc_split["test_label"], data_field="Data", labels_field="class")
	train_net.train(batch_size=200)