import os
import h5py
import shutil
import pandas as pd

class H5Utilities(object):
	@staticmethod
	def createFile(data_as_np, write_path):
		with h5py.File(write_path, 'w') as f:
			f.create_dataset('Data', data=data_as_np, chunks=True, dtype='f')
		return

	@staticmethod
	def read(data_path, data_field):
		return h5py.File(data_path, 'r')[data_field]

class CsvUtilities(object):
	@staticmethod
	def createFile(pd_dataframe, write_path, header_list, index=False):
		pd_dataframe.to_csv(write_path, header=header_list, index=index)

	@staticmethod
	def read(data_path, class_field):
		return pd.read_csv(data_path)[class_field]

	@staticmethod
	def toPdDataframe(data):
		return pd.DataFrame(data, columns=['class'])

class NetLogger(object):
	@staticmethod
	def logTraining(log_path, train_size, train_pos_count, test_size, test_pos_count, dropout):
		with open(log_path, 'w') as f:
			f.write("Training size: {0}\n".format(train_size))
			f.write("Training nodule count: {0}\n".format(train_pos_count))
			f.write("Test size: {0}\n".format(test_size))
			f.write("Test nodule count: {0}\n".format(test_pos_count))
			f.write("Dropout: {0}\n".format(dropout))

class DirectoryManager(object):
	@staticmethod
	def checkExists(path):
		return os.path.isdir(path)

	@staticmethod
	def delete(path):
		if(DirectoryManager.checkExists(path)):
			shutil.rmtree(path)
		return

	@staticmethod
	def create(path, dir_name, force=False):
		# If force, will wipe out the directory if it exists. Else, an error will be raised
	 	dir_path 	= os.path.join(path, dir_name)
	 	
	 	if(DirectoryManager.checkExists(dir_path)):
	 		if(not force):
	 			raise Exception("Directory exits and cannot create a new one")
 			DirectoryManager.delete(dir_path)

 		os.makedirs(dir_path)
 		return dir_path