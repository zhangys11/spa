# The code is partly based on https://github.com/LixiangHan/GANs-for-1D-Signal
import os
import pandas as pd
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.datasets as dataset
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TorchDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx]
        sample = torch.from_numpy(sample)
        return sample

def train_wgan(X, noise_dim = 100, 
               clip_value = 0.1, lr = 1e-4, 
               epochs=64, batch_size=8, output_dir = './output',
               verbose = True):
    
    # 当生成网络G训练1个batch时，判别网络D要接着训练n_critic次，即n_critic个batch。
    # 因为WGAN对判别网络D做的约束太强，需要给它多一些学习的机会。
    n_critic = 3 # default 5

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    trainloader = torch.utils.data.DataLoader(
        TorchDataset(X), batch_size=batch_size, shuffle=True
    )
    
    # init netD and netG
    netD = Discriminator(feature_dim=X.shape[1]).to(device)
    #fake_input = torch.randn(1, 1, X.shape[1], device=device)
    #fake_output = netD(fake_input)
    #print(fake_output.shape)
    #noise_dim = fake_output.shape[1]
    netD.apply(weights_init)
 
    netG = Generator(noise_dim, feature_dim=X.shape[1]).to(device)
    netG.apply(weights_init)

    # used for visualizing training process
    fixed_noise = torch.randn(16, noise_dim, 1, device=device)

    # optimizers
    optimizerD = optim.RMSprop(netD.parameters(), lr=lr)
    optimizerG = optim.RMSprop(netG.parameters(), lr=lr)

    for epoch in range(epochs):
        for step, data in enumerate(trainloader):

            # training netD
            data = torch.unsqueeze(data.to(device), 1).float()
            b_size = data.size(0)
            netD.zero_grad()

            noise = torch.randn(b_size, noise_dim, 1, device=device)
            fake = netG(noise)

            loss_D = -torch.mean(netD(data)) + torch.mean(netD(fake))
            loss_D.backward()
            optimizerD.step()

            for p in netD.parameters():
                # print(p.data.max(), p.data.min())
                p.data.clamp_(-clip_value, clip_value)

            if step % n_critic == 0:
                # train netG
                noise = torch.randn(b_size, noise_dim, 1, device=device)

                netG.zero_grad()
                fake = netG(noise)
                loss_G = -torch.mean(netD(fake))

                netD.zero_grad()
                netG.zero_grad()
                loss_G.backward()
                optimizerG.step()
            
                if verbose:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                        % (epoch, epochs, step+1, len(trainloader), 
                           loss_D.item(), loss_G.item()))

        if verbose:
            # save generated samples along the training process
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
                
                f, a = plt.subplots(4, 4, figsize=(16, 8))
                f.suptitle('epoch'+str(epoch))
                for i in range(4):
                    for j in range(4):
                        a[i][j].plot(fake[i * 4 + j].view(-1))
                        a[i][j].set_xticks(())
                        a[i][j].set_yticks(())
                plt.savefig(output_dir + '/wgan_epoch_%d.png' % epoch)
                plt.close()
                # plt.show()

    return netG, netD
    # save model
    # torch.save(netG, './wgan_netG.pkl')
    # torch.save(netD, './wgan_netD.pkl')

def expand_dataset(X, y, nobs, X_names=None, 
                   epochs=500, batch_size=16, noise_dim=100,
                   verbose = True):
    '''
    use WGAN () to upsample. 
    Generate equal number of samples for each class.

    nobs : how many samples to generate.
    verbose: when True, will output generator's architecture graph.

    Remarks
    -------
    Wasserstein GAN (WGAN) 的核心在于采用Wasserstein距离（也称为Earth Mover's Distance，EMD）
    替代传统GAN中的Jensen-Shannon散度作为判别器的损失函数。
    Wasserstein距离衡量的是两个概率分布之间的“推土机成本”，
    它在概率分布差异较小或不完全重叠时仍能提供有意义的梯度信息。

    Wasserstein GAN (WGAN) 相比原始GAN的算法实现流程只改了四点：
    1. 判别器最后一层去掉sigmoid
    2. 生成器和判别器的loss不取log
    3. 每次更新判别器的参数之后把它们的绝对值截断到不超过一个固定常数c
    4. 不要用基于动量的优化算法（包括momentum和Adam），推荐RMSProp，SGD也行
    '''
    epochs = max(epochs, 200) # we found WGAN needs at least several hundred epochs for spectroscopic data

    if not X_names:
        X_names = list(range(X.shape[1]))
        
    synth_data = pd.DataFrame()

    plot_once = True

    for label in set(y):
        data = pd.concat([pd.DataFrame(X), pd.DataFrame(y, columns=['label'])], axis=1)
        data = data[data['label'] == label]
        Xi = data.iloc[:, :-1]
        yi = data.iloc[:, -1]
        Xi.columns = Xi.columns.astype(float)
        
        g, d = train_wgan(Xi.values, noise_dim=noise_dim, 
                          clip_value = 0.1, lr = 1e-3,
                          epochs=epochs, batch_size=batch_size, 
                          verbose = verbose)
        
        noise = torch.randn(nobs, noise_dim, 1, device=device)
        fake = g(noise).squeeze(1).cpu().detach().numpy()
        
        df = pd.DataFrame(fake, columns = X_names)
        df['label'] = len(df) * [label]
        synth_data = pd.concat([synth_data, df], axis=0)

        if verbose and plot_once:
            
            from torchviz import make_dot
            import IPython.display

            input_vec = torch.randn(1, noise_dim, 1, device=device)
            IPython.display.display('<b>WGAN generator</b>')
            IPython.display.display(make_dot(g(input_vec)))

            plot_once = False # only need draw once

            # IPython.display.display('<b>WGAN discriminator</b>')
            # vec = g(input_vec)
            # IPython.display.display(make_dot(d(vec)))

    synth_data.reset_index(drop=True,inplace=True)
    return synth_data.iloc[:, :-1], synth_data.iloc[:, -1] #, generator_plot, discriminator_plot


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Discriminator(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 以上几步，不断减半输出维度，得到 (256, signal_dim / 8) 的尺寸
            nn.Conv1d(256, 1, kernel_size=feature_dim // 8, stride=1, padding=0, bias=False),
        )

    def forward(self, x, y=None):
        x = self.layers(x)
        return x

class Generator(nn.Module):
    def __init__(self, noise_dim, feature_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose1d(noise_dim, 256, feature_dim // 8, 1, 0, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.ConvTranspose1d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.ConvTranspose1d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.ConvTranspose1d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh(),

            # 以上几步与D对称相反，不断加倍输出维度，得到 (1, noise_dim * 8) 的尺寸。
            # 即使设置 noise_dim = signal_dim //8，由于无法保证noise_dim * 8严格等于signal_dim （如 100*8 != 801），
            # 因此加一个linear做尺寸调整

            nn.Linear(feature_dim // 8 * 8, feature_dim) # this needs further inspection
        )

    def forward(self, x):
        x = self.layers(x)
        return x