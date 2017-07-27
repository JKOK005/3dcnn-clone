import h5py
import pandas as pd
import numpy as np 
import random as rd
from UtilitiesScript import *
from settings import params

class DatasetGenerator(object):
	def __init__(self):
		self.pos_data 	= None
		self.neg_data 	= None

	def __stitch(self, data_A, class_A, data_B, class_B):
		data 		= np.concatenate([data_A, data_B], axis=0)
		A_to_np 	= np.full((len(data_A), 1), class_A, dtype=int)
		B_to_np 	= np.full((len(data_B), 1), class_B, dtype=int)
		class_tmp 	= np.concatenate([A_to_np, B_to_np], axis=0)
		labels 		= pd.DataFrame(class_tmp, columns=['class'])
		return [data, labels]

	def __split(self, full_data, ratio):
		print("Begin splitting of data")
		full_len 	= len(full_data)
		smpl_len 	= int(ratio *full_len)

		# train_indx 	= rd.sample(range(full_len), k=smpl_len)
		# test_indx 	= list(set(range(full_len)) - set(train_indx))

		training 	= full_data[0:smpl_len]
		testing 	= full_data[smpl_len:]
		return [training, testing]

	def load(self, pos_dir, neg_dir, data_field):
		print("Positive nodule path: {0}".format(pos_dir))
		print("Negative nodule path: {0}".format(neg_dir))
		self.pos_data 	= h5py.File(pos_dir, 'r')[data_field]
		self.neg_data 	= h5py.File(neg_dir, 'r')[data_field]
		return 

	def setGenerator(self, ratio):
		assert ratio > 0 and ratio < 1
		assert self.pos_data is not None and self.neg_data is not None
		[pos_train, pos_test] = self.__split(self.pos_data, ratio)
		[neg_train, neg_test] = self.__split(self.neg_data, ratio)

		[train_data, train_labels] 	= self.__stitch(neg_train, 0, pos_train, 1)
		[test_data, test_labels] 	= self.__stitch(neg_test, 0, pos_test, 1)

		train_set 	= {"data": train_data, "labels": train_labels}
		test_set 	= {"data": test_data, "labels": test_labels}
		return [train_set, test_set]

if __name__ == "__main__":
	dsetGen = DatasetGenerator()
	dsetGen.load(params["sc_split"]["pos_nodules"], params["sc_split"]["neg_nodules"], "Data")
	[train_set, test_set] = dsetGen.setGenerator(0.8)

	# Write data files to storage directory
	H5Utilities.createFile(train_set['data'], params["sc_split"]["train_data"])
	H5Utilities.createFile(test_set['data'], params["sc_split"]["test_data"])

	# Write label files to storage directory
	CsvUtilities.createFile(train_set['labels'], params["sc_split"]["train_label"], header_list=["class"])
	CsvUtilities.createFile(test_set['labels'], params["sc_split"]["test_label"], header_list=["class"])
	