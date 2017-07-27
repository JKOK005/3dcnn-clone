import os
import numpy as np
from UtilitiesScript import *
from settings import params

class NetworkStats(object):
	@staticmethod
	def __consoleLog(true_pos_count, false_pos_count, false_neg_count, true_neg_count, precision, recall, f1):
		print("True positive count: {0}".format(true_pos_count))
		print("False positive count: {0}".format(false_pos_count))
		print("False negative count: {0}".format(false_neg_count))
		print("True negative count: {0}".format(true_neg_count))

		print("Precision: {0}".format(precision))
		print("Recall: {0}".format(recall))
		print("F1: {0} \n".format(f1))

	@staticmethod
	def report(predicted, ground_truth, thresh, disp=True):
		pred_pos 		= np.where(predicted >= thresh)
		true_nodules 	= np.where(ground_truth == 1)

		true_pos 		= np.intersect1d(pred_pos, true_nodules)
		false_pos 		= np.setxor1d(pred_pos, true_pos)
		false_neg 		= np.setxor1d(true_nodules, true_pos)

		sample_count 	= len(predicted)
		true_pos_count 	= len(true_pos)
		false_pos_count = len(false_pos)
		false_neg_count = len(false_neg)
		true_neg_count 	= sample_count - true_pos_count - false_pos_count - false_neg_count

		precision 		= true_pos_count / (true_pos_count + false_pos_count + 0.0)
		recall 			= true_pos_count / (true_pos_count + false_neg_count + 0.0)
		f1 				= 2*precision*recall / (precision + recall)
		false_pos_rate 	= false_pos_count / (false_pos_count + true_neg_count)
		
		if(disp):
			NetworkStats.__consoleLog(true_pos_count, false_pos_count, false_neg_count, true_neg_count, precision, recall, f1)
		return

if __name__ == "__main__":
	model_dir 		= "set_p_6228_n_31140_raw_data_aug"			# Model directory where the evaluated .csv file is stored	
	file_str 		= "predicted_03-0.0421-0.0596.hd5.csv" 			# Name of the fiile inside the model directory

	result_path 	= os.path.join(params["sc_test"]["results_dir"], model_dir, file_str)
	predicted  		= CsvUtilities.read(result_path, class_field="class")
	ground_truth 	= CsvUtilities.read(params["backup"]["label"], class_field="class")
	
	# threshold 		= 0.8 	# Between 0 - 1
	for threshold in np.arange(0.01, 1, 0.01):
		print("Threshold {0}".format(threshold))
		NetworkStats.report(predicted, ground_truth, threshold, disp=True)
