import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter






class Generator (nn.Module):

	def __init__(self, numZ = 100, numGenFeat = 64, colorChannels = 1):
		super(Generator, self).init()

		#The number of out channels is different from the original paper, because:
		#1) The dataset we use is not in an RGB format, instad it only contains grayscale Images
		#2) The image size is lower than it was in the original paper
		#In order to make the network more easily adaptable, we parameterized the in and out-sizes
		# as well as the size of the random input vector and the number of color channnels
		self.gen = nn.Sequential(
			#First layer: Input of size numZ goes into first convolution
			#Stride = 2, Padding = 1???
			nn.ConvTranspose2d(in_channels = numZ, out_channels = numGenFeat * 8,kernel_size = 4, stride = 1, padding = 0, bias=False),
			nn.BatchNorm2d(numGenFeat * 8),
			nn.ReLU(inplace = True),
			#
			#
			nn.ConvTranspose2d(in_channels=numGenFeat * 8, out_channels=numGenFeat * 4, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(numGenFeat * 4),
			nn.ReLU(inplace=True),
			#
			#
			nn.ConvTranspose2d(in_channels=numGenFeat * 4, out_channels=numGenFeat * 2, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(numGenFeat * 2),
			nn.ReLU(inplace=True),
			#
			#
			nn.ConvTranspose2d(in_channels=numGenFeat * 2 , out_channels=numGenFeat, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(numGenFeat),
			nn.ReLU(inplace=True),
			#
			#
			nn.ConvTranspose2d(in_channels = numGenFeat,  out_channels = colorChannels , kernel_size = 4 , stride = 2, padding = 1, bias=False),
			nn.Tanh()
		)

		def forward(self, input):
			return self.gen(input)

class Discriminator(nn.Module):

	def __init__(self, numZ = 100, numDiscFeat = 64, colorChannels = 1, wasserstein = False):
		super(Discriminator, self).__init__()
		if not wasserstein:
			self.disc = nn.Sequential(
				nn.ConvTranspose2d(in_channels = numDiscFeat, out_channels = colorChannels, kernel_size = 4, stride = 2, padding = 1, bias = False),
				nn.LeakyReLU(0.2, inplace=True),
				#
				#
				nn.Conv2d(in_channels = numDiscFeat, out_channels = numDiscFeat * 2, kernel_size = 4, stride = 2, padding = 1, bias=False),
				nn.BatchNorm2d(numDiscFeat * 2),
				nn.LeakyReLU(0.2, inplace=True),
				# state size. (ndf*2) x 16 x 16
				nn.Conv2d(in_channels = numDiscFeat * 2, out_channels = numDiscFeat * 4, kernel_size = 4, stride = 2, padding = 1, bias=False),
				nn.BatchNorm2d(ndf * 4),
				nn.LeakyReLU(0.2, inplace=True),
				# state size. (ndf*4) x 8 x 8
				nn.Conv2d(in_channels = numDiscFeat * 4, out_channels = numDiscFeat * 8, kernel_size = 4, stride = 2, padding = 1, bias=False),
				nn.BatchNorm2d(numDiscFeat * 8),
				nn.LeakyReLU(0.2, inplace=True),
				# state size. (ndf*8) x 4 x 4
				nn.Conv2d(in_channels = numDiscFeat * 8, out_channels = 1, kernel_size = 4, stride = 1, padding = 0, bias=False),
				nn.Sigmoid()
			)
		else:
			self.disc = nn.Sequential(
				nn.ConvTranspose2d(in_channels=numDiscFeat, out_channels=colorChannels, kernel_size=4, stride=2,
								   padding=1, bias=False),
				nn.LeakyReLU(0.2, inplace=True),
				#
				#
				nn.Conv2d(in_channels=numDiscFeat, out_channels=numDiscFeat * 2, kernel_size=4, stride=2, padding=1,
						  bias=False),
				nn.BatchNorm2d(numDiscFeat * 2),
				nn.LeakyReLU(0.2, inplace=True),
				# state size. (ndf*2) x 16 x 16
				nn.Conv2d(in_channels=numDiscFeat * 2, out_channels=numDiscFeat * 4, kernel_size=4, stride=2, padding=1,
						  bias=False),
				nn.BatchNorm2d(ndf * 4),
				nn.LeakyReLU(0.2, inplace=True),
				# state size. (ndf*4) x 8 x 8
				nn.Conv2d(in_channels=numDiscFeat * 4, out_channels=numDiscFeat * 8, kernel_size=4, stride=2, padding=1,
						  bias=False),
				nn.BatchNorm2d(numDiscFeat * 8),
				nn.LeakyReLU(0.2, inplace=True),
				# state size. (ndf*8) x 4 x 4
				nn.Conv2d(in_channels=numDiscFeat * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
				nn.Sigmoid()
			)

	def forward(self, input):
		return self.dis(input)

def init_weights(n):
    #Initialize the weights, as done in the DCGan Paper
    #Quote from the paper:
    #"All weights were initialized from a zero-centered Normal distribution with standard deviation 0.02"
    center  = 0
    stDev   = 0.02
