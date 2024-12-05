from modelDropout import PolicyGeneratorUnet1D,PolicyDiscriminatorCB
import time
import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from dataset import Dataset
from policy_generator.common.pytorch_util import dict_apply
from policy_generator.model.common.normalizer import LinearNormalizer
import wandb 
import os 

SEED = 43
print('seed: ', SEED)
torch.manual_seed(SEED)

writer = SummaryWriter('runs/'+str(time.time()))

BATCH_SIZE = 128
LR = 2e-4
BETAS = (0.5,0.999)
LATENT_SIZE = 2*1024
TRAJ_SIZE = 128
TASK_SIZE = 128
EPOCHES = 5000
CHECKPOINT = 100
WEIGHT_DECAY = 1e-5
RESUME_TRAINING = False
WEIGHTS_PATH = " "
WEIGHT_PATH_RESUME = ""
PROJECT_NAME = 'AMRL'

device = 'cuda:0'

try:
    os.makedirs(WEIGHTS_PATH)
    print('Created dir:',WEIGHTS_PATH)
except:
    print('Already created dir:',WEIGHTS_PATH)

# wandb.init(
#     # set the wandb project where this run will be logged
#     project=PROJECT_NAME,
#     # track hyperparameters and run metadata
#     config={
#     "learning_rate": LR,
#     "architecture": "AMRL",
#     "dataset": "",
#     "epochs": EPOCHES,
#     "weight_decay":WEIGHT_DECAY,
#     "batch_size":BATCH_SIZE,
#     "latent_size":LATENT_SIZE,
#     "trajectory_size":TRAJ_SIZE,
#     "task_size":TASK_SIZE,
#     "checkpoints":CHECKPOINT,
#     "weight_decay":WEIGHT_DECAY,
#     "resume_training":RESUME_TRAINING,
#     "weights_path":WEIGHTS_PATH,
#     "weight_path_resume":WEIGHT_PATH_RESUME,
#     "seed":SEED,
#     "device":device
#     }
# )

discriminator = PolicyDiscriminatorCB(LATENT_SIZE,TRAJ_SIZE).to(device)
generator = PolicyGeneratorUnet1D(2*TRAJ_SIZE,LATENT_SIZE).to(device)

generator_optimizer = torch.optim.Adam(generator.parameters(),betas=BETAS,lr = LR,weight_decay=WEIGHT_DECAY)

# discriminator_optimizer = torch.optim.SGD(discriminator.parameters(),
#                                         lr = LR,momentum=0.9,
#                                         nesterov=True,
#                                         weight_decay=WEIGHT_DECAY)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(),betas=BETAS,lr = LR,weight_decay=WEIGHT_DECAY)

mi_optimizer = torch.optim.Adam([{'params':generator.parameters()},{'params':discriminator.parameters()}],
                                betas=BETAS,
                                lr = LR,
                                weight_decay=WEIGHT_DECAY)

if RESUME_TRAINING:
    checkpoint = torch.load(WEIGHT_PATH_RESUME)
    discriminator.load_state_dict(checkpoint['policy_discriminator'])
    generator.load_state_dict(checkpoint['policy_generator'])
    generator_optimizer.load_state_dict(checkpoint['generator_optimizer'])
    discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer'])
    mi_optimizer.load_state_dict(checkpoint['mi_optimizer'])
    print('Resume weights!')

dataset = Dataset() 
normalizer = dataset.get_normalizer()
train_dataloader = dataset.train_dataloader()

traj_loss_fn = torch.nn.MSELoss()
bce_fn = torch.nn.BCELoss()
task_loss_fn = torch.nn.MSELoss()
photometric_loss = torch.nn.MSELoss()


if __name__ == '__main__':
    i = 0
    range_list = range(EPOCHES)
    if RESUME_TRAINING:
        checkpoint = torch.load(WEIGHT_PATH_RESUME)
        EPOCHES = checkpoint['epoches']
        range_list = range(checkpoint['current_epoch'],EPOCHES)
        i = checkpoint['iteration']
        
    for e in range_list:
        for batch_idx, batch in enumerate(train_dataloader):
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            
            batch = normalizer.normalize(batch)
            traj_data = batch['traj'].to(device)
            task_data = batch['task'].to(device)
            parms_data = batch['param'].reshape(BATCH_SIZE,-1).to(device) # 128 x 2048
            noise = torch.randn((BATCH_SIZE,TRAJ_SIZE)).to(device)
            
            # Discriminator Network
            discriminator_optimizer.zero_grad()
            real,_ = discriminator(parms_data)
            episilon_real =  torch.distributions.Uniform(torch.tensor([0.01]),torch.tensor([0.2])).sample(real.shape).to(device).squeeze(-1)
            real_loss = bce_fn(real,(torch.ones_like(real).to(device)-episilon_real).detach()) 
            z = torch.cat((noise,traj_data),dim=1)
            out_fake_z = generator(z)
            fake,_ = discriminator(out_fake_z.detach())
            episilon_fake =  torch.distributions.Uniform(torch.tensor([0.01]),torch.tensor([0.2])).sample(fake.shape).to(device).squeeze(-1)
            fake_loss = bce_fn(fake,(torch.zeros_like(fake).to(device)+episilon_fake).detach())
            discriminator_loss = 0.5*(fake_loss+real_loss)
            discriminator_loss.backward()
            discriminator_optimizer.step()
            writer.add_scalar('Discriminator Loss',discriminator_loss.item(),i)
            writer.add_scalar('Real Loss',real_loss.item(),i)
            writer.add_scalar('Fake Loss',fake_loss.item(),i)
            writer.add_scalar('Batch_size',BATCH_SIZE,i)
            writer.add_scalar('Learning Rate',LR,i)
            loss_dict = dict()
            loss_dict = {'discriminator_loss':discriminator_loss.item(),
                        'real_loss':real_loss.item(),
                        'fake_loss':fake_loss.item()}
            
            # Generator Network
            generator_optimizer.zero_grad()
            for _ in range(2):
                z = torch.cat((noise,traj_data),dim=1)
                out_z = generator(z)
                out_flag,traj_out = discriminator(out_z)
                episilon_real =  torch.distributions.Uniform(torch.tensor([0.01]),torch.tensor([0.3])).sample(out_flag.shape).to(device).squeeze(-1)
                gen_loss = bce_fn(out_flag,(torch.ones_like(out_flag).to(device)-episilon_real).detach()) + 1e3*photometric_loss(out_z,parms_data)
                gen_loss.backward()
                generator_optimizer.step()
            writer.add_scalar('Generator Loss',gen_loss.item(),i)
            loss_dict['generator_loss']= gen_loss.item()
            
            # MI Regularization
            mi_optimizer.zero_grad()
            z = torch.cat((noise,traj_data),dim=1)
            out_z = generator(z)
            out_flag,traj_out = discriminator(out_z)
            mi_loss = 1e3*(traj_loss_fn(traj_out,traj_data))
            mi_loss.backward()
            mi_optimizer.step()
            writer.add_scalar('MI Loss',mi_loss.item(),i)
            loss_dict['mi_loss'] = mi_loss.item()
            
            
            if (i+1)%10 == 0:
                print('Epoches:',e+1,
                        'Generator Loss:',gen_loss.item(),
                        'MI Loss:',mi_loss.item(),
                        'Discriminator Loss:',discriminator_loss.item(),
                        'Real Loss:',real_loss.item(),
                        'Fake Loss:',fake_loss.item())
                # wandb.log(loss_dict)
            i += 1
        if (e+1)%CHECKPOINT == 0:
            torch.save({'policy_generator':generator.state_dict(),
                        'policy_discriminator':discriminator.state_dict(),
                        'normalizer':normalizer.state_dict(),
                        'generator_optimizer':generator_optimizer.state_dict(),
                        'discriminator_optimizer':discriminator_optimizer.state_dict(),
                        'mi_optimizer':mi_optimizer.state_dict(),
                        'current_epoch':e+1,
                        'iteration':i,
                        'epoches':EPOCHES,
                        'weights_path':os.path.join(WEIGHTS_PATH,'checkpoint'+str(e+1)+'_seed'+str(SEED)+'.pth')},
                        os.path.join(WEIGHTS_PATH,'checkpoint'+str(e+1)+'_seed'+str(SEED)+'.pth'))
            print('Model weights Saved!')
    # wandb.finish()