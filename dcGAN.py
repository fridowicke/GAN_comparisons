import torch.nn as nn


class Generator(nn.Module):

	def __init__(self, numZ=100, numGenFeat=64, colorChannels=1, Wasserstein=False):
		super(Generator, self).__init__()

		# The number of out channels is different from the original paper, because:
		# 1) The dataset we use is not in an RGB format, instead it only contains grayscale Images
		# 2) The image size is lower than it was in the original paper
		# In order to make the network more easily adaptable, we parameterized the in and out-sizes
		# as well as the size of the random input vector and the number of color channnels
		if not Wasserstein:
			self.gen = nn.Sequential(

				nn.ConvTranspose2d(in_channels=numZ, out_channels=numGenFeat * 16, kernel_size=4, stride=1, padding=0,
								   bias=False),
				nn.ReLU(),  # inplace= True),
				#
				#
				nn.ConvTranspose2d(in_channels=numGenFeat * 16, out_channels=numGenFeat * 8, kernel_size=4, stride=2,
								   padding=1, bias=False),
				nn.ReLU(),  # inplace=True),
				#
				#
				nn.ConvTranspose2d(in_channels=numGenFeat * 8, out_channels=numGenFeat * 4, kernel_size=4, stride=2,
								   padding=1, bias=False),
				nn.ReLU(),  # inplace=True),
				#
				#
				nn.ConvTranspose2d(in_channels=numGenFeat * 4, out_channels=numGenFeat * 2, kernel_size=4, stride=2,
								   padding=1, bias=False),
				nn.ReLU(),  # inplace=True),
				#
				#
				nn.ConvTranspose2d(in_channels=numGenFeat * 2, out_channels=colorChannels, kernel_size=4, stride=2,
								   padding=1, bias=False),
				nn.Tanh()
			)
		else:
			self.gen = nn.Sequential(
				nn.ConvTranspose2d(in_channels=numZ, out_channels=numGenFeat * 16, kernel_size=4, stride=1, padding=0,
								   bias=False),
				nn.BatchNorm2d(numGenFeat * 16),
				nn.ReLU(),
				#
				#
				nn.ConvTranspose2d(in_channels=numGenFeat * 16, out_channels=numGenFeat * 8, kernel_size=4, stride=2,
								   padding=1, bias=False),
				nn.BatchNorm2d(numGenFeat * 8),
				nn.ReLU(),  # inplace=True),
				#
				#
				nn.ConvTranspose2d(in_channels=numGenFeat * 8, out_channels=numGenFeat * 4, kernel_size=4, stride=2,
								   padding=1, bias=False),
				nn.BatchNorm2d(numGenFeat * 4),
				nn.ReLU(),  # inplace=True),
				#
				#
				nn.ConvTranspose2d(in_channels=numGenFeat * 4, out_channels=numGenFeat * 2, kernel_size=4, stride=2,
								   padding=1, bias=False),
				nn.BatchNorm2d(numGenFeat * 2),
				nn.ReLU(),  # inplace=True),
				#
				#
				nn.ConvTranspose2d(in_channels=numGenFeat * 2, out_channels=colorChannels, kernel_size=4, stride=2,
								   padding=1, bias=False),
				nn.Tanh()
			)

	def forward(self, input):
		return self.gen(input)


class Discriminator(nn.Module):

	def __init__(self, numZ=100, numDiscFeat=64, colorChannels=1, wasserstein=False):
		super(Discriminator, self).__init__()
		if not wasserstein:
			self.disc = nn.Sequential(
				#
				nn.Conv2d(in_channels=colorChannels, out_channels=numDiscFeat, kernel_size=4, stride=2, padding=1,
						  bias=False),
				nn.LeakyReLU(0.2),  # , inplace=True),
				#
				#
				nn.Conv2d(in_channels=numDiscFeat, out_channels=numDiscFeat * 2, kernel_size=4, stride=2, padding=1,
						  bias=False),
				nn.BatchNorm2d(numDiscFeat * 2),
				nn.LeakyReLU(0.2),  # , inplace=True),
				#
				nn.Conv2d(in_channels=numDiscFeat * 2, out_channels=numDiscFeat * 4, kernel_size=4, stride=2, padding=1,
						  bias=False),
				nn.BatchNorm2d(numDiscFeat * 4),
				nn.LeakyReLU(0.2),  # , inplace=True),
				#
				nn.Conv2d(in_channels=numDiscFeat * 4, out_channels=numDiscFeat * 8, kernel_size=4, stride=2, padding=1,
						  bias=False),
				nn.BatchNorm2d(numDiscFeat * 8),
				nn.LeakyReLU(0.2),  # , inplace=True),
				#
				nn.Conv2d(in_channels=numDiscFeat * 8, out_channels=1, kernel_size=4, stride=2, padding=0, bias=False),
				nn.Sigmoid()
			)
		else:
			self.disc = nn.Sequential(
				nn.Conv2d(in_channels=colorChannels, out_channels=numDiscFeat, kernel_size=4, stride=2, padding=1,
						  bias=False),
				nn.LeakyReLU(0.2),  # , inplace=True),
				#
				#
				nn.Conv2d(in_channels=numDiscFeat, out_channels=numDiscFeat * 2, kernel_size=4, stride=2, padding=1,
						  bias=False),
				nn.BatchNorm2d(numDiscFeat * 2),
				nn.LeakyReLU(0.2),  # , inplace=True),
				nn.Conv2d(in_channels=numDiscFeat * 2, out_channels=numDiscFeat * 4, kernel_size=4, stride=2, padding=1,
						  bias=False),
				nn.BatchNorm2d(numDiscFeat * 4),
				nn.LeakyReLU(0.2),  # , inplace=True),
				nn.Conv2d(in_channels=numDiscFeat * 4, out_channels=numDiscFeat * 8, kernel_size=4, stride=2, padding=1,
						  bias=False),
				nn.BatchNorm2d(numDiscFeat * 8),
				nn.LeakyReLU(0.2),  # , inplace=True),
				nn.Conv2d(in_channels=numDiscFeat * 8, out_channels=1, kernel_size=4, stride=2, padding=0, bias=False),
				# nn.Sigmoid()
			)

	def forward(self, input):
		return self.disc(input)


def weightInit(model):
	# Initialize the weights, as done in the DCGan Paper
	# Quote from the paper:
	# "All weights were initialized from a zero-centered Normal distribution with standard deviation 0.02"
	center = 0
	stDev = 0.02
	for mod in model.modules():
		if (isinstance(mod, (nn.ConvTranspose2d, nn.Conv2d, nn.BatchNorm2d))):
			nn.init.normal(mod.weight.data, center, stDev)

