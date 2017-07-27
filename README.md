## 3D CNN for Lung Nodule Detection
### Author: Johan Kok Zhi Kang

This is a cloned repository. The network architecture has been removed owing to confidentiality issues.
This repository only displays the scripts needed to setup and train the 3D neural network. 
___


This repository contains the scripts needed to train a CNN against the Luna 16 data set. 

The Luna 16 data set consists of 888 patient images in Dicom format. These images stitched into 3D volumes,
reshaped into a suitable dimension and output into .h5 format. Hence, no further processing of the raw images
is needed. 

To train and evaluate the network, the following sequence is executed
1. The dataset is split into both positive and negative subsets
2. The subsets are merged and thereafter split into training and validation subsets on a specified ratio
3. The network is trained on the training set and validated on the validation set
4. Evaluate is performed on the test dataset which is the entire Luna 16 sample volumes
5. Network statistics such as F1 score / Precision / Recall are obtained to assess the network


### Data
Raw Luna 16 dataset can be downloaded via this [link](https://luna16.grand-challenge.org/download/). First, ensure that you have an account 
and have registered for the challenge. 

On the server side, the preprocessed data is located under the directory: /data/saturn_home/zhangww/johan/kaggle_source/network_v2_devel/data/full_dataset.

This directory contains
1. save_patch_raw.h5 				- Raw lung volumes in 32 x 32 x 32 voxel dimensions
2. save_patch_raw_reshaped.h5 		- Reshape of (**1**) for compatibility with inputs to the neural net
3. candidates_v2.csv 				- Candidate labels for each volume entry in (**1**), as downloaded from the Luna 16 database.
										- These labels are used for training the net. 


### Execution
Ensure that your current working director is in the **scripts** folder.

```
cd scripts
```

In this folder, first edit *settings.py* with suitable parameters

The *settings.py* script exports global parameters in a Python dictionary format. These parameters
specify file locations and storage spaces. To add a new configuration to the setting, create a dictionary
and add it to the *__export_list* list as such

```python
__export_list 		+= [{
	"sc_test" 		:
	{
		"results_dir" 			: __results_dir,
	}
}]
```

Thereafter, ensure that you have your positive and negative data stored in a suitable location under the *data* folder.
Add this folder to your settings parameter and run the following command. 

```python
python DatasetGenerator
```

This will generate the training and validation datasets in the same folder. 

We can then proceed to train the network. The network structure is located in the *NetController.py* script. The *get_net* 
interface in the script allows the user to implement his own network which will be called by subsequent training scripts. 

Training is done using the following command

```python
python TrainNetwork
```

The user can configure various training parameters such as the number of epochs, batch size and dropout activation by changing the script.

By default, training is done using the [Adam](https://keras.io/optimizers/) optimizer to perform gradient steps to achieve a global minima of the loss function. 
The loss function chosen is the [Binary Crossenthropy](https://en.wikipedia.org/wiki/Cross_entropy). 

Checkpoints per epoch is implement to store the result after each epoch (configuration accessible via params -> sc_train -> checkpoint_opt). 
Each checkpoint is saved as a .hd5 file under the *trained_weights* directory with the title

```
{epoch}-{loss}-{val_loss}
```

In addition a log file is stored in the same director. This file reports the status of the dataset used to train the network. 

Once the network has been trained, we can evaluate it using 

```python
python EvaluateNetwork
```

There are 2 ways to evaluate a trained network. 
1. Evaluation based on all epochs (**evaluateNetAll**) 			- We evaluate each checkpoint model weights against the training dataset
2. Evaluate the best performing model (**evaluateNetBest**) 	- We evaluate only the best performing model with lowest validation loss

Be sure to set the right parameters for the (model_dir) parameter in the script. The parameters should specify the file name under
the *trained_weights* directory to which the models are stored. 

The results are stored as a .csv extension in the *results* directory. 

Finally, we can gain insights into the performance of the network using

```python
python NetworkStats
```

Ensure that the (model_dir) and (file_str) variable are configured to point to the results dir / .csv file name under the *results* folder.


### Additional scripts 
1. Augment.py 			- Used to augment incoming data via random rotations and translations along each axis
2. UtilitiesScript.py 	- Contains functionalities common to all scripts, such as reading .h5 files, creating .csv files or making directories
3. visualize2D.py 		- Visualization of lung volume data


### Dependencies
The main dependency is Keras. Please ensure that Keras is installed with **Tensorflow** backend. **Do not use Teano** unless you want to go
inside the main scripts and change the configurations.

Optimally, performance can be optimized using GPU parallel processing. Keras will automatically detect any available GPU that we authorize it to use.
Configuration of the environment variable is necessary through

```
export CUDA_VISIBLE_DEVICES="1,2,3..."
```