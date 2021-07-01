import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

# The following is an implementation of the original Generative adversarial Network
# proposed by Goodfellow et.al. in 2014.
#

# Design the model
class Discriminator(nn.Module)

	def __init__(self):

		super().__init__()
		#We use a simple Sequential Neural network with one hidden layer
		#The LeakyReLu activation function was used to avoid the dying ReLu problem
		# which is described in https://himanshuxd.medium.com/activation-functions-sigmoid-relu-leaky-relu-and-softmax-basics-for-neural-networks-and-deep-8d9c70eed91e
		#The Sigmoid function in the end is used to ensure the final output is between zero and one
		self.crit 	= nn.Sequential( nn.Linear(inSize, 64), nn.LeakyReLU(0.01), nn.Linear(64, 1), nn.Sigmoid() )
		self.opt 	= optim.Adam(, lr=lr)

model = Sequential()
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
