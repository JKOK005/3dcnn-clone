import os

cur_dir 		= os.getcwd()
__export_list 	= []

__data_dir 		= os.path.join(cur_dir, '..', 'data')

''' Configuration for Augmenting positive nodules '''
__store_dir 		= os.path.join(__data_dir, 'set_p_650000_n_753418')
__pos_to_aug 		= os.path.join(__store_dir, 'positive_raw_data.h5')
__pos_nodule_path 	= os.path.join(__store_dir, 'augmented_pos_raw_data.h5')

__export_list 		+= [{ 
	"sc_aug" : 
	{
		"model_dir" 	: __store_dir,
		"pos_raw" 		: __pos_to_aug,
		"pos_nodules"	: __pos_nodule_path,
	},
}]

''' End of section '''


''' Configuration for splitting data into training and testing subsets '''

__neg_nodule_path 	= os.path.join(__store_dir, 'negative_raw_data.h5')
__train_data_path 	= os.path.join(__store_dir, 'training_file.h5')
__train_label_path 	= os.path.join(__store_dir, 'training_labels.csv')
__test_data_path 	= os.path.join(__store_dir, 'testing_file.h5')
__test_label_path 	= os.path.join(__store_dir, 'testing_labels.csv')

__export_list 		+= [{ 
	"sc_split" : 
	{
		"model_dir" 	: __store_dir,
		"pos_nodules" 	: __pos_nodule_path,
		"neg_nodules"	: __neg_nodule_path,
		"train_data" 	: __train_data_path,
		"train_label" 	: __train_label_path,
		"test_data" 	: __test_data_path,
		"test_label" 	: __test_label_path,
	},
}]

''' End of section '''


''' Configuration for raw backup data '''

__bkup_dir 			= os.path.join(__data_dir, "full_dataset")
__raw_data_bkup 	= os.path.join(__bkup_dir, "save_patch_raw_reshaped.h5")
__raw_labels_bkup 	= os.path.join(__bkup_dir, "candidates_V2.csv")

__export_list 		+= [{
	"backup" 	: 
	{
		"data" 			: __raw_data_bkup,
		"label" 		: __raw_labels_bkup,
	}		
}]

''' End of section '''



''' Configuration for training network '''

__model_main_dir 	= os.path.join(cur_dir, '..', 'trained_weights')
__log_name 			= "log.txt"
__checkpoint_opt 	= "{epoch:02d}-{loss:.4f}-{val_loss:.4f}.hd5"

__export_list 		+= [{
	"sc_train"  	: 
	{
		"model_main_dir" 	: __model_main_dir,
		"log_name" 			: __log_name,
		"checkpoint_opt" 	: __checkpoint_opt,
	}
}]

''' End of section '''


''' Configuration for testing network '''

__results_dir 		= os.path.join(cur_dir, '..', 'results')

__export_list 		+= [{
	"sc_test" 		:
	{
		"results_dir" 			: __results_dir,
	}
}]

''' End section '''

# Export settings
params 		= {}
for __each in __export_list:
	params.update(__each)