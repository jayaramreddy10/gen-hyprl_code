from modelDropout import PolicyGeneratorUnet1D
from transformer_encoder_residue import SampleLatentNet
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

# writer = SummaryWriter('runs/'+str(time.time()))

BATCH_SIZE = 128
TOTAL_SAMPLES = 8
LR = 2e-4
BETAS = (0.5,0.999)
LATENT_SIZE = 2*1024
TRAJ_SIZE = 128
TASK_SIZE = 128
EPOCHES = 3000
CHECKPOINT = 50
WEIGHT_DECAY = 1e-5
LAMBDA = 1e3
RESUME_TRAINING = False
WEIGHTS_PATH = "sample_latent_net_res_xyz"
WEIGHT_PATH_RESUME = "weights_best/checkpoint3400_seed43.pth"
PROJECT_NAME = 'AMRL'

device = 'cuda:0'

try:
    os.makedirs(WEIGHTS_PATH)
    print('Created dir:',WEIGHTS_PATH)
except:
    print('Already created dir:',WEIGHTS_PATH)

smpl_net = SampleLatentNet(LATENT_SIZE).to(device)

generator = PolicyGeneratorUnet1D(2*TRAJ_SIZE,LATENT_SIZE).to(device)
checkpoint = torch.load(WEIGHT_PATH_RESUME)
generator.load_state_dict(checkpoint['policy_generator'])
generator.eval()


smpl_optimizer = torch.optim.Adam(smpl_net.parameters(),betas=BETAS,lr = LR,weight_decay=WEIGHT_DECAY)


dataset = Dataset() 
normalizer = dataset.get_normalizer()
normalizer.load_state_dict(checkpoint['normalizer'])
normalizer.eval()
train_dataloader = dataset.train_dataloader()

latent_loss_fn = torch.nn.SmoothL1Loss()
aux_loss_fn = torch.nn.MSELoss()
cosine_sim_loss_gn = torch.nn.CosineSimilarity(dim=1, eps=1e-6)


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
            traj_data = batch['traj'].detach().to(device)
            task_data = batch['task'].to(device)
            gt_parms_data = batch['param'].reshape(BATCH_SIZE,-1).to(device) # 128 x 2048
            
            i += 1
            
            list_noisy_latent = list()
            # list_noisy_traj = list()
            list_res = list()
            for j in range(TOTAL_SAMPLES):
                noise = torch.randn((BATCH_SIZE,TRAJ_SIZE)).to(device)
                with torch.no_grad():
                    z = torch.cat((noise,traj_data),dim=1)
                    latent = generator(z)
                    list_noisy_latent.append(latent.unsqueeze(1))
                    list_res.append(gt_parms_data.unsqueeze(1)-latent.unsqueeze(1))
                    # list_noisy_traj.append(z.unsqueeze(1))
            
            noisy_latents = torch.cat(list_noisy_latent,dim=1)
            noisy_traj = traj_data.unsqueeze(1)
            gt_res = torch.cat(list_res,dim=1)
            gt_res = gt_res.mean(1)
            
            smpl_optimizer.zero_grad()
            out_latents,out_res = smpl_net(noisy_latents,noisy_traj)
            loss_latent = LAMBDA*latent_loss_fn(out_res,gt_res.detach())
            loss_cs = LAMBDA*(1.-cosine_sim_loss_gn(out_latents,gt_parms_data)).mean()
            loss_aux = LAMBDA*(aux_loss_fn(out_latents+out_res,gt_parms_data))
            loss = loss_latent + loss_cs + loss_aux
            loss.backward()
            smpl_optimizer.step()
            if (i%100)==0:
                print('Epoch:',e+1,'Iterations:',i,'Loss:',loss.item(),'loss res:',loss_latent.item(),'Loss_cs:',loss_cs.item(),'Loss_aux:',loss_aux.item())
        if (e+1)%CHECKPOINT == 0:
            torch.save({'samplenet':smpl_net.state_dict(),
                        'optimizer':smpl_optimizer.state_dict(),
                        'current_epoch':e+1,
                        'iteration':i,
                        'epoches':EPOCHES,
                        'weights_path':os.path.join(WEIGHTS_PATH,'checkpoint'+str(e+1)+'_seed'+str(SEED)+'.pth')},
                        os.path.join(WEIGHTS_PATH,'checkpoint'+str(e+1)+'_seed'+str(SEED)+'.pth'))
            print('Model weights Saved!')
