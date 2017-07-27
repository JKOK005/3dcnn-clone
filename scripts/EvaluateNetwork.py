import os
import glob
from NetController import NetController
from UtilitiesScript import *
from settings import params

class EvaluateNet(object):
	def __init__(self, trained_model_dir, test_data_path, name_str):
		self.model_dir 		= trained_model_dir
		self.test_data 		= H5Utilities.read(test_data_path, data_field="Data")
		self.name_str 		= name_str

	def __savePrediction(self, pd_dataframe, file_name):
		DirectoryManager.create(params["sc_test"]["results_dir"], self.name_str, force=True)
		save_model_dir 	= os.path.join(params["sc_test"]["results_dir"], self.name_str, file_name)
		CsvUtilities.createFile(pd_dataframe, save_model_dir, header_list=["class"])
		return

	def evaluateNet(self, net, batch_size, verbose=1):
		return net.predict(self.test_data, batch_size=batch_size, verbose=verbose)

	def evaluateNetBest(self, batch_size, dropout, verbose=1, save_pred=True):
		all_models 		= glob.glob(self.model_dir + '/*.hd5')
		min_loss 		= 999

		for each_model in all_models:
			model_name          = each_model.split('/')[-1]
			model_rm_extension  = model_name[:-len(".hd5")]
			val_loss            = model_rm_extension.split('-')[-1]
			if(float(val_loss) < min_loss):
				best_performing_model   = each_model
				min_loss                = float(val_loss)

		net 			= NetController.get_net(load_weight_path=best_performing_model, USE_DROPOUT=dropout)
		print("Evaluating: {0}".format(best_performing_model))
		res 			= self.evaluateNet(net, batch_size=batch_size, verbose=verbose)
		pred 			= res[:,1]
		pd_dataframe 	= CsvUtilities.toPdDataframe(pred)
		if(save_pred):
			file_name 	= best_performing_model.split('/')[-1]	
			self.__savePrediction(pd_dataframe, file_name)

	def evaluateNetAll(self, batch_size, dropout, verbose=1, save_pred=True):
		all_models 	= glob.glob(self.model_dir + '/*.hd5')

		for each_model in all_models:
			net 			= NetController.get_net(load_weight_path=each_model, USE_DROUPUT=dropout)
			res 			= self.evaluateNet(net, batch_size=batch_size, verbose=verbose)
			pred 			= res[:,1]
			pd_dataframe 	= CsvUtilities.toPdDataframe(pred)
			if(save_pred):
				file_name 	= each_model.split('/')[-1]	
				self.__savePrediction(pd_dataframe, file_name)


if __name__ == "__main__":
	model_dir 			= "July_20_2017_16_17_13"
	weights_dir 		= params["sc_train"]["model_main_dir"]
	trained_model_dir 	= os.path.join(weights_dir, model_dir)

	test_data_path		= params["backup"]["data"]
	evaluator 			= EvaluateNet(trained_model_dir, test_data_path, model_dir)
	evaluator.evaluateNetBest(batch_size=200, dropout=True, verbose=1, save_pred=True)
