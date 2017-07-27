from keras.layers import *
from keras.layers.convolutional import ZeroPadding3D
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model, Sequential
from keras.optimizers import Adam
from keras.metrics import *

class NetController(object):

	@classmethod
	def __model_JKOK005(cls, input_shape=(32, 32, 32, 1), load_weight_path=None, USE_DROPOUT=False):
		# Network is removed due to confidentiality issues.

	@classmethod
	def get_net(cls, load_weight_path=None, USE_DROPOUT=None):
		return cls.__model_JKOK005(load_weight_path=load_weight_path, USE_DROPOUT=USE_DROPOUT) 		# Implement and change the return model for a new set of model

if __name__ == "__main__":
	NetController.get_net()