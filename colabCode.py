import torch.nn as nn
import torch
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils


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


# Create the Training parameters

batchSize = 128  # DCGan paper uses Bath size of 128
imageSize = 64  # FashionMNIST: 16x16
colorChannels = 1  # Fashin Mnist: one (grayscale)
numZ = 100
numGenFeat = 64
numDiscFeat = 64
numEpochs = 5
learningRate = 0.0002
numWorkers = 2
k = 5

# Load the Fashion - Mnist Data from Torchvision
# Normalize it and turn it into a Tensor

# transformation = transforms.Compose([transforms.Resize(imageSize), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])
transformation = transforms.Compose(
    [transforms.Resize(imageSize), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ])
data = datasets.FashionMNIST(root="dataset/", transform=transformation, download=True)
loader = DataLoader(data, batch_size=batchSize, shuffle=True)  # , num_workers = numWorkers)

# Set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def trainWasserstein(k=3, lr=0.0002, nZ=100, ngf=64, ndf=64):
    numZ = nZ
    numGenFeat = ngf
    numDiscFeat = ndf
    learningRate = lr

    gen = Generator(numZ, numGenFeat, colorChannels, True).to(device)
    disc = Discriminator(numZ, numDiscFeat, colorChannels, wasserstein=True).to(device)
    weightInit(gen)
    weightInit(disc)

    optGen = optim.RMSprop(gen.parameters(), lr=learningRate)
    optDisc = optim.RMSprop(disc.parameters(), lr=learningRate)

    orgImg = SummaryWriter(
        f"logs/lr={learningRate}numZ={numZ}numGenFeat={numGenFeat}numDiscFeat={numDiscFeat}/originalImagesWs")
    genImg = SummaryWriter(
        f"logs/lr={learningRate}numZ={numZ}numGenFeat={numGenFeat}numDiscFeat={numDiscFeat}/generatedImagesWs")
    loss = SummaryWriter(f"logs/lr={learningRate}numZ={numZ}numGenFeat={numGenFeat}numDiscFeat={numDiscFeat}/lossWs")
    step = 0

    for epoch in range(numEpochs):
        for count, (org, _) in enumerate(loader):

            org = org.to(device)
            batchSize = org.shape[0]

            # Training of the Discriminator
            # If Wasserstein == False, then k=1
            for _ in range(k):
                noise = torch.randn(imageSize, numZ, 1, 1, device=device)
                generated = gen(noise)
                # print(org.shape, "----------------_ORGSHAPE_")
                discData = disc(org).reshape(-1)
                discGen = disc(generated).reshape(-1)
                discLoss = -(torch.mean(discData) - torch.mean(discGen))
                disc.zero_grad()
                discLoss.backward(retain_graph=True)
                optDisc.step()

                for p in disc.parameters():
                    # p.data.clamp_(-0.01, 0.01)
                    p.data.clamp(-0.01, 0.01)

            genFake = disc(generated).reshape(-1)
            genLoss = -torch.mean(genFake)
            gen.zero_grad()
            genLoss.backward()
            optGen.step()

            # Printing to Tensorboard
            if count % 100 == 0:
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


def trainDC(metric="KL", lr=0.0002, nZ=100, ngf=64, ndf=64):
    numZ = nZ
    numGenFeat = ngf
    numDiscFeat = ndf
    learningRate = lr

    gen = Generator(numZ, numGenFeat, colorChannels).to(device)
    disc = Discriminator(100, numDiscFeat, colorChannels).to(device)
    weightInit(gen)
    weightInit(disc)

    optGen = optim.Adam(gen.parameters(), lr=learningRate, betas=(0.5, 0.999))
    optDisc = optim.Adam(disc.parameters(), lr=learningRate, betas=(0.5, 0.999))

    metric = nn.BCELoss()
    if metric == "KL":
        metric = nn.KLDivLoss()

    imgNoise = torch.randn(32, numZ, 1, 1).to(device)
    orgImg = SummaryWriter(
        f"logs/lr={learningRate}numZ={numZ}numGenFeat={numGenFeat}numDiscFeat={numDiscFeat}/originalImagesDC")
    genImg = SummaryWriter(
        f"logs/lr={learningRate}numZ={numZ}numGenFeat={numGenFeat}numDiscFeat={numDiscFeat}/generatedImagesDC")
    loss = SummaryWriter(f"logs/lr={learningRate}numZ={numZ}numGenFeat={numGenFeat}numDiscFeat={numDiscFeat}/loss")
    step = 0

    gen.train()
    disc.train()

    for epoch in range(numEpochs):
        for count, (org, _) in enumerate(loader):
            org = org.to(device)
            noise = torch.randn(batchSize, numZ, 1, 1).to(device)
            generated = gen(noise)

            #Train the Discriminator
            discData = disc(org).reshape(-1)
            discDataLoss = metric(discData, torch.ones_like(discData))
            discGen = disc(generated.detach()).reshape(-1)
            discGenLoss = metric(discGen, torch.zeros_like(discGen))
            # discLoss = -(1 / 2) * (metric(discData, torch.ones_like(discData)) + metric(discGen, torch.zeros_like(discGen)))
            discLoss = (1 / 2) * (discDataLoss + discGenLoss)
            disc.zero_grad()
            discLoss.backward()
            optDisc.step()

            #Train the Generator
            decision = disc(generated).reshape(-1)
            genLoss = metric(decision, torch.ones_like(decision))
            gen.zero_grad()
            genLoss.backward()
            optGen.step()

            # Printing to Tensorboard in every one hundredth step
            if count % 100 == 0:
                print(
                    f"Epoch [{epoch}/{numEpochs}] Batch {count}/{len(loader)} \
                      Loss D: {discLoss:.4f}, loss G: {genLoss:.4f}"
                )

                with torch.no_grad():
                    fake = gen(imgNoise)
                    img_grid_real = torchvision.utils.make_grid(
                        org[:32], normalize=True
                    )
                    img_grid_fake = torchvision.utils.make_grid(
                        fake[:32], normalize=True
                    )

                    orgImg.add_image("Real", img_grid_real, global_step=step)
                    genImg.add_image("Fake", img_grid_fake, global_step=step)
                    loss.add_scalar("Total loss", (-discLoss + genLoss), count)

                step += 1


def testParams(lr=True, featGen=True, featDisc=True, numZ=True):
    # lr = True
    # featGen = False
    # featDisc = False
    # numZ = False
    learningRate = 0.0002

    if lr:

        for i in range(7):
            lr = learningRate * 10 ** (i - 3)
            trainDC(lr=lr)
            trainWasserstein(lr=lr)

    if numZ:
        vals = [1, 10, 100, 1000]
        for z in vals:
            trainDC(nZ=z)
            trainWasserstein(nZ=z)

    if featGen:
        feats = [16, 32, 64, 128]
        for nfg in feats:
            trainDC(ngf=nfg)
            trainWasserstein(ngf=nfg)

    if featDisc:
        feats = [16, 32, 64, 128]
        for nfd in feats:
            trainDC(ndf=nfd)
            trainWasserstein(ndf=nfd)