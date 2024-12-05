import torch
import torch.nn as nn
import torchvision
from policy_generator.model.common.normalizer import LinearNormalizer

class PolicyGeneratorUnet1D(torch.nn.Module):
    def __init__(self,input_size,latent_size):
        super(PolicyGeneratorUnet1D,self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.main = torch.nn.Sequential(
            torch.nn.Linear(self.input_size,1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024,1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024,1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024,self.latent_size),
        )
        self.d1 = torch.nn.Linear(self.input_size,self.latent_size)
        self.d2 = torch.nn.Sequential(torch.nn.Linear(self.latent_size,1024),
                                      torch.nn.BatchNorm1d(num_features=1024),
                                      torch.nn.GELU(),
                                      torch.nn.Dropout(0.1))
        self.d3 = torch.nn.Sequential(torch.nn.Linear(1024,512),
                                      torch.nn.BatchNorm1d(num_features=512),
                                      torch.nn.GELU(),
                                      torch.nn.Dropout(0.1))
        self.d4 = torch.nn.Sequential(torch.nn.Linear(512,256),
                                      torch.nn.BatchNorm1d(num_features=256),
                                      torch.nn.GELU(),
                                      torch.nn.Dropout(0.1))
        
        self.u3 = torch.nn.Sequential(torch.nn.Linear(256,512),
                                      torch.nn.BatchNorm1d(num_features=512),
                                      torch.nn.GELU(),
                                      torch.nn.Dropout(0.1))
        
        self.u2 = torch.nn.Sequential(torch.nn.Linear(512,1024),
                                      torch.nn.BatchNorm1d(num_features=1024),
                                      torch.nn.GELU(),
                                      torch.nn.Dropout(0.1))
        self.u1 = torch.nn.Linear(1024,2048)
        # self.normalizer = LinearNormalizer()
    def forward(self,x):
        d1 = self.d1(x) # 2048
        d2 = self.d2(d1) #1024
        d3 = self.d3(d2) #512
        d4 = self.d4(d3) #256
        u3 = self.u3(d4)+d3 #512
        u2 = self.u2(u3)+d2 #1024
        u1 = self.u1(u2)+d1 #2028
        return u1

class PolicyGenerator(torch.nn.Module):
    def __init__(self,input_size,latent_size):
        super(PolicyGenerator,self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.main = torch.nn.Sequential(
            torch.nn.Linear(self.input_size,1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024,1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024,1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024,self.latent_size),
        )
        # self.normalizer = LinearNormalizer()
    def forward(self,x):
        x = self.main(x)
        return x

class PolicyDiscriminator(torch.nn.Module):
    def __init__(self,latent_size,traj_dim,task_dim):
        super(PolicyDiscriminator,self).__init__()
        self.latent_size = latent_size
        self.traj_dim = traj_dim
        self.task_dim = task_dim
        self.main = torch.nn.Sequential(
            torch.nn.Linear(self.latent_size,1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024,1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024,512),
            torch.nn.GELU(),
        )
        self.discriminator_fc = torch.nn.Sequential(torch.nn.Linear(512,1),
            torch.nn.Sigmoid())
        self.traj_fc = torch.nn.Linear(512,self.traj_dim)
        self.task_fc = torch.nn.Linear(512,self.task_dim)
    
    def forward(self,x):
        x = self.main(x)
        discriminator_out = self.discriminator_fc(x)
        traj_out = self.traj_fc(x)
        task_out = self.task_fc(x)
        return discriminator_out,traj_out,task_out

class PolicyDiscriminatorCB(torch.nn.Module):
    def __init__(self,latent_size,traj_dim):
        super(PolicyDiscriminatorCB,self).__init__()
        self.latent_size = latent_size
        self.traj_dim = traj_dim
        self.main = torch.nn.Sequential(
            torch.nn.Linear(self.latent_size,1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024,1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024,512),
            torch.nn.GELU(),
        )
        self.discriminator_fc = torch.nn.Sequential(torch.nn.Linear(512,1),
            torch.nn.Sigmoid())
        self.traj_fc = torch.nn.Linear(512,self.traj_dim)
    
    def forward(self,x):
        x = self.main(x)
        discriminator_out = self.discriminator_fc(x)
        traj_out = self.traj_fc(x)
        return discriminator_out,traj_out


if __name__ == '__main__':
    batch_size = 128
    noise_dim = 128
    traj_dim = 128 
    task_dim = 128
    disc_input_latent = 2*1024
    discriminator = PolicyDiscriminator(latent_size=disc_input_latent,traj_dim=traj_dim,task_dim=task_dim)
    discriminator_out,traj_out,task_out = discriminator(torch.randn(batch_size,disc_input_latent))
    print('Policy Discriminator Output Size:',discriminator_out.shape)
    print('Policy Discriminator Trajectory Size:',traj_out.shape)
    print('Policy Discriminator Task Size:',task_out.shape)
    generator = PolicyGenerator(input_size=3*noise_dim,latent_size=disc_input_latent)
    generator_out = generator(torch.randn(batch_size,3*noise_dim))
    print('Policy Generator Output Size:',generator_out.shape)
    generator = PolicyGeneratorUnet1D(input_size=3*noise_dim,latent_size=disc_input_latent)
    generator_out = generator(torch.randn(batch_size,3*noise_dim))
    print('Policy Generator Unet1D Output Size:',generator_out.shape)
