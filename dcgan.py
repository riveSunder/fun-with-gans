import argparse
import os
import random
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
#import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#from IPython.display import HTML

# ensure repeatability
my_seed = 1
random.seed(my_seed)
torch.manual_seed(my_seed)

dataroot = "images"
num_workers = 2
batch_size = 1024 
image_size = 64 
num_channels = 3
dim_z = 64 

gen_features = 64
disc_lr = 1e-4
gen_lr = 2e-4
beta1 = 0.5
beta2 = 0.999
num_epochs = 1000
ngpu = 2


def weights_init(my_model):
    classname = my_model.__class__.__name__

    if classname.find("Conv") != -1:
        nn.init.normal_(my_model.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(my_model.weight.data, 1.0, 0.02)
        nn.init.constant_(my_model.bias.data, 0.0)


dataset = dset.ImageFolder(root=dataroot,\
            transform=transforms.Compose([\
            transforms.Resize(image_size),\
            transforms.CenterCrop(image_size),\
            transforms.ToTensor(),\
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),\
        ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,\
                shuffle=True, num_workers=num_workers)

device = torch.device("cuda:0" if ngpu > 0 and torch.cuda.is_available() else "cpu")
print(device)

# look at batch of real images
real_batch = next(iter(dataloader))
def disp_batch_grid(real_batch):
    plt.figure(figsize=(10,10))
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64],\
            padding=2, normalize=True).cpu(), (1,2,0)))
    plt.show()  


class Generator(nn.Module):

    def __init__(self, ngpu, dim_z, gen_features, num_channels):
        super(Generator, self).__init__()

        self.ngpu = ngpu
        self.main = nn.Sequential(\
                nn.ConvTranspose2d(dim_z, gen_features*8, 4, 1, 0, bias=False),\
                nn.BatchNorm2d(gen_features*8),\
                nn.ReLU(True),\
                nn.ConvTranspose2d(gen_features*8, gen_features*4, 4, 2, 1, bias=False),\
                nn.BatchNorm2d(gen_features*4),\
                nn.ReLU(True),\
                nn.ConvTranspose2d(gen_features*4, gen_features*2, 4, 2, 1, bias=False),\
                nn.BatchNorm2d(gen_features*2),\
                nn.ReLU(True),\
                nn.ConvTranspose2d(gen_features*2, gen_features, 4, 2, 1, bias=False),\
                nn.BatchNorm2d(gen_features),\
                nn.ReLU(True),\
                nn.ConvTranspose2d(gen_features, num_channels, 4, 2, 1, bias=False),\
                nn.Tanh()
            )
                
    def forward(self, z):
        return self.main(z)

class Discriminator(nn.Module):

    def __init__(self, ngpu, gen_features, num_channels):
        super(Discriminator, self).__init__()

        self.ngpu = ngpu
        self.main = nn.Sequential(\
                nn.Conv2d(num_channels, gen_features, 4, 2, 1, bias=False),\
                nn.LeakyReLU(0.2, True),\
                nn.Conv2d(gen_features, gen_features*2, 4, 2, 1, bias=False),\
                nn.BatchNorm2d(gen_features*2),\
                nn.LeakyReLU(0.2, True),\
                nn.Conv2d(gen_features*2, gen_features*4, 4, 2, 1, bias=False),\
                nn.BatchNorm2d(gen_features*4),\
                nn.LeakyReLU(0.2, True),\
                nn.Conv2d(gen_features*4, gen_features*8, 4, 2, 1, bias=False),\
                nn.BatchNorm2d(gen_features*8),\
                nn.LeakyReLU(0.2, True),\
                nn.Conv2d(gen_features*8, 1, 4, 1, 0, bias=False),\
                nn.Sigmoid()\
            )
                
    def forward(self, imgs):
        return self.main(imgs)

gen_net = Generator(ngpu, dim_z, gen_features, \
        num_channels).to(device)        

disc_net = Discriminator(ngpu, gen_features, num_channels).to(device)

# add data parallel here for >= 2 gpus
if (device.type == "cuda") and (ngpu > 1):
    disc_net = nn.DataParallel(disc_net, list(range(ngpu)))
    gen_net = nn.DataParallel(gen_net, list(range(ngpu)))

gen_net.apply(weights_init)
disc_net.apply(weights_init)

criterion = nn.BCELoss()

fixed_noise = torch.randn(64, dim_z, 1, 1, device=device)

real_label = 1
fake_label = 0

disc_optimizer = optim.Adam(disc_net.parameters(), lr=disc_lr, betas=(beta1, beta2))
gen_optimizer = optim.Adam(gen_net.parameters(), lr=gen_lr, betas=(beta1, beta2))

img_list = []
gen_losses = []
disc_losses = []
iters = 0

t0 = time.time()

for epoch in range(num_epochs):
    for ii, data in enumerate(dataloader,0):
        
        # update the discriminator
        disc_net.zero_grad()

        # discriminator pass with real images 
        real_cpu = data[0].to(device)
        batch_size= real_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=device)
        output = disc_net(real_cpu).view(-1)
        disc_real_loss = criterion(output,label)
        disc_real_loss.backward()

        disc_x = output.mean().item()
        
        # discriminator pass with fake images
        noise = torch.randn(batch_size, dim_z, 1, 1, device=device)

        fake = gen_net(noise)
        label.fill_(fake_label)

        output = disc_net(fake.detach()).view(-1)

        disc_fake_loss = criterion(output, label)

        disc_fake_loss.backward()

        disc_gen_z1 = output.mean().item()

        disc_loss = disc_real_loss + disc_fake_loss

        disc_optimizer.step()

        # update the generator
        gen_net.zero_grad()
        label.fill_(real_label)
        output = disc_net(fake).view(-1)

        gen_loss = criterion(output, label)

        gen_loss.backward()

        disc_gen_z2 = output.mean().item()

        gen_optimizer.step()

        if ii % 100 == 0:
            print("{} {:.3f} s |Epoch {}/{}:\tdisc_loss: {:.3e}\tgen_loss: {:.3e}\tdisc(x): {:.3e}\tdisc(gen(z)): {:.3e}/{:.3e}".format(iters,time.time()-t0, epoch, num_epochs, disc_loss.item(), gen_loss.item(), disc_x, disc_gen_z1, disc_gen_z2))

        disc_losses.append(disc_loss.item())
        gen_losses.append(gen_loss.item())

        if (iters % 500 == 0) or \
                ((epoch == num_epochs-1) and (ii == len(dataloader)-1)):

            with torch.no_grad():
                fake = gen_net(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True).numpy())

            np.save("./gen_images.npy", img_list)
            np.save("./gen_losses.npy", gen_losses)
            np.save("./disc_losses.npy", disc_losses)
            torch.save(gen_net.state_dict(), "./weights/generator.h5")
            torch.save(disc_net.state_dict(), "./weights/discriminator.h5")
        iters += 1

