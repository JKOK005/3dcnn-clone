import h5py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.ndimage.interpolation import *
from settings import params

if __name__ == "__main__":
	data 			= h5py.File(params["sc_split"]["test_data"],'r')['Data']
	print(len(data))

	for indx in range(0, 100):
		print(data[indx].shape)
		vol_segment 	= data[indx].reshape(32,32,32)
		img_viewer 		= plt.imshow(vol_segment[0], cmap=plt.cm.gray)
		
		for i in range(len(vol_segment)):
			vol 		= vol_segment[i]
			img_viewer.set_data(vol)
			plt.pause(0.01)  
