import h5py
import pandas as pd
import numpy as np
import random
from settings import params
from datetime import datetime
from scipy.ndimage.interpolation import *

class VolAugmenter(object):
	def __init__(self, rand_seed):
		random.seed(rand_seed)

	def translate(self, vol, vox_seq=[1.0,1.0,1.0]):
		vol 		= vol.reshape((32,32,32))
		trans_vol 	= shift(vol, shift=vox_seq, mode="nearest")
		return trans_vol.reshape(32,32,32,1)

	def rotate(self, vol, deg):
		vol 	= vol.reshape((32,32,32))
		rot_vol = rotate(vol, angle=deg, axes=(1,2), reshape=False, mode="nearest")
		return rot_vol.reshape(32,32,32,1)

	def augment(self, vol):
		vox_seq 	= [random.uniform(-1,1) for i in range(3)]
		deg 		= random.randint(0,3) *90 + random.uniform(-5,5)

		trans_vol	= self.translate(vol, vox_seq=vox_seq)
		final_vol 	= self.rotate(vol, deg=deg)
		return final_vol

if __name__ == "__main__":
	data 		= h5py.File(params["sc_aug"]["pos_raw"], 'r')['Data']
	data_size 	= len(data)
	img_size 	= 650000
	augmenter 	= VolAugmenter(datetime.now())

	with h5py.File(params["sc_aug"]["pos_nodules"], 'w') as f:
		dset 	= f.create_dataset('Data', (img_size,32,32,32,1), dtype='f')
		for i in range(img_size):
			dset[i] = augmenter.augment(data[i % data_size])