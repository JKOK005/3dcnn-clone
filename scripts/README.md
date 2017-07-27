
### CNN training scripts
This is the main folder which houses essential scripts needed to run and train the neural net. 

In addition to the core scripts mentioned in the previous page, the user is free to use a couple of utility scripts

1. Augment.py 			- Generates randomized augmentations on a dataset.

To edit the number of generated dataset, set the parameters
```python
img_size 	= (No. of datasets)
```

2. UtilitiesScript.py 	- Contains common utility functions that with classes that handles
..* H5Utilities 		- H5py read and write 
..* CsvUtilities 		- Csv read / write for pandas dataframe
..* NetLogger 			- Logging for each new training instance of the network
..* DirectoryManager 	- Management of directory creation and deletion 


3. visualize2D.py 		- Visualization of data in .h5 format. Each volume is displayed in lapses of 2D data in a plot. 