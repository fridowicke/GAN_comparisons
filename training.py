import torch
import torch.optim as optim
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
k               = 5


#Load the Fashion - Mnist Data from Torchvision
#Normalize it and turn it into a Tensor

transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])
data = datasets.FashionMNIST(root="dataset/", transform = transformation, download=True)
loader = DataLoader(data, batch_size=batchSize, shuffle=True) #, num_workers = numWorkers)


#Set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def trainWasserstein(k = 5):

    gen         = gan.Generator(numZ, numGenFeat, colorChannels).to(device)
    disc        = gan.Discriminator(numZ, numDiscFeat, colorChannels, wasserstein = True)

    optGen     = optim.RMSprop(gen.parameters(), lr = learningRate)
    optDisc    = optim.RMSprop(disc.parameters(), lr = learningRate)

    orgImg     = SummaryWriter(f"logs/originalImages")
    genImg     = SummaryWriter(f"logs/generatedImages")
    loss        = SummaryWriter(f"logs/loss")
    step = 0

    for epoch in range(numEpochs):
        for count, (org,_) in enumerate(loader):

            org     = org.to(device)
            batchSize = org.shape[0]

            #Training of the Discriminator
            #If Wasserstein == False, then k=1
            for _ in range(k):
                noise = torch.randn(imageSize, numZ, 1, 1, device=device)
                generated = gen(noise)
                print(org.shape, "----------------_ORGSHAPE_")
                discData = disc(org).reshape(-1)
                discGen  = disc(generated).reshape(-1)
                discLoss = -(torch.mean(discData) - torch.mean(discGen))
                disc.zero_grad()
                discLoss.backward(retain_graph=True)
                optDisc.step()

                for p in disc.parameters():
                    #p.data.clamp_(-0.01, 0.01)
                    p.data.clamp(-0.01, 0.01)

            genFake = disc(generated).reshape(-1)
            genLoss = -torch.mean(genFake)
            gen.zero_grad()
            genLoss.backward()
            optGen.step()

            #Printing to Tensorboard
            if count % 10 == 0:
                gen.eval()
                disc.eval()
                print(
                    f"Epoch [{epoch}/{numEpochs}] Batch {count}/{len(loader)} \
                              Loss D: {discLoss:.4f}, loss G: {genLoss:.4f}"
                )

                with torch.no_grad():
                    fake = gen(noise)
                    # take out (up to) 32 examples
                    img_grid_real = torchvision.utils.make_grid(
                        org[:32], normalize=True
                    )
                    img_grid_fake = torchvision.utils.make_grid(
                        generated[:32], normalize=True
                    )

                    orgImg.add_image("Real", img_grid_real, global_step=step)
                    genImg.add_image("Fake", img_grid_fake, global_step=step)
                    loss.add_scalar("Total loss", discLoss + genLoss, count)

                step += 1
                gen.train()
                disc.train()

def trainDC(k = 1):

    gen         = gan.Generator(numZ, numGenFeat, colorChannels).to(device)
    disc        = gan.Discriminator(numZ, numDiscFeat, colorChannels, wasserstein = False)

    optGen     = optim.RMSprop(gen.parameters(), lr = learningRate)
    optDisc    = optim.RMSprop(disc.parameters(), lr = learningRate)

    orgImg     = SummaryWriter(f"logs/originalImages")
    genImg     = SummaryWriter(f"logs/generatedImages")
    loss        = SummaryWriter(f"logs/loss")
    step = 0

    for epoch in range(numEpochs):
        for count, (org,_) in enumerate(loader):

            org     = org.to(device)
            batchSize = org.shape[0]

            #Training of the Discriminator
            #If Wasserstein == False, then k=1
            for _ in range(k):
                noise = torch.randn(imageSize, numZ, 1, 1, device=device)
                generated = gen(noise)
                print(org.shape, "----------------_ORGSHAPE_")
                discData = disc(org).reshape(-1)
                discGen  = disc(generated).reshape(-1)
                discLoss = -(torch.mean(discData) - torch.mean(discGen))
                disc.zero_grad()
                discLoss.backward(retain_graph=True)
                optDisc.step()

                for p in disc.parameters():
                    #p.data.clamp_(-0.01, 0.01)
                    p.data.clamp(-0.01, 0.01)

            genFake = disc(generated).reshape(-1)
            genLoss = -torch.mean(genFake)
            gen.zero_grad()
            genLoss.backward()
            optGen.step()

            #Printing to Tensorboard
            if count % 10 == 0:
                gen.eval()
                disc.eval()
                print(
                    f"Epoch [{epoch}/{numEpochs}] Batch {count}/{len(loader)} \
                              Loss D: {discLoss:.4f}, loss G: {genLoss:.4f}"
                )

                with torch.no_grad():
                    fake = gen(noise)
                    # take out (up to) 32 examples
                    img_grid_real = torchvision.utils.make_grid(
                        org[:32], normalize=True
                    )
                    img_grid_fake = torchvision.utils.make_grid(
                        generated[:32], normalize=True
                    )

                    orgImg.add_image("Real", img_grid_real, global_step=step)
                    genImg.add_image("Fake", img_grid_fake, global_step=step)
                    loss.add_scalar("Total loss", discLoss + genLoss, count)

                step += 1
                gen.train()
                disc.train()

def testPlot():
    #This function plots datastes to test the dataloader
    #Not sure if it works with colored images
    batch = next(iter(loader))
    plt.figure(figsize=(8,8))
    plt.imshow(np.transpose(torchvision.utils.make_grid(batch[0].to(device)[:16], normalize=True).cpu(),(1,2,0)))
    plt.show()