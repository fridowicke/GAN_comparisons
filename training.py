import torch
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils
import matplotlib.pyplot as plt
import dcGAN as gan
import numpy as np



#Create the Training parameters

batchSize       = 128 #DCGan paper uses Bath size of 128
imageSize       = 64  #FashionMNIST: 16x16
colorChannels   = 1   #Fashin Mnist: one (grayscale)
numZ            = 100
numGenFeat      = 64
numDiscFeat     = 64
numEpochs       = 5
learningRate    = 0.0002
numWorkers      = 2


#Load the Fashion - Mnist Data from Torchvision
#Normalize it and turn it into a Tensor

transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])
data = datasets.FashionMNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(data, batch_size=batchSize, shuffle=True)


#Set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gen     = gan.Generator(numZ, numGenFeat, colorChannels).to(device)
disc    = gan.Discriminator(numZ, numDiscFeat, colorChannels, wasserstein = False)




def testPlot():
    #This function plots datastes to test the dataloader
    #Not sure if it works with colored images
    batch = next(iter(loader))
    plt.figure(figsize=(8,8))
    plt.imshow(np.transpose(torchvision.utils.make_grid(batch[0].to(device)[:16], normalize=True).cpu(),(1,2,0)))
    plt.show()