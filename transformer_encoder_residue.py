import torch
import torch.nn as nn
import torchvision

class SampleLatentNet(torch.nn.Module):
    def __init__(self,emb_size):
        super(SampleLatentNet,self).__init__()
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, emb_size))
        self.linear_proj_1 = torch.nn.Sequential(torch.nn.Linear(emb_size,256),
                                            torch.nn.GELU(),
                                            torch.nn.Dropout(0.2),
                                            torch.nn.Linear(256,256))
        self.traj_proj = torch.nn.Linear(128,256)
        self.linear_proj_2 = torch.nn.Sequential(torch.nn.Linear(256,256),
                                            torch.nn.GELU(),
                                            torch.nn.Dropout(0.2),
                                            torch.nn.Linear(256,256))
        self.encoder = torch.nn.TransformerEncoderLayer(d_model=256, nhead=128)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder, num_layers=1)
        self.proj_out = torch.nn.Sequential(torch.nn.Linear(256,256),
                                            torch.nn.GELU(),
                                            torch.nn.Dropout(0.2),
                                            torch.nn.Linear(256,256))
        self.latent = torch.nn.Sequential(torch.nn.Linear(256,256),
                                          torch.nn.GELU(),
                                          torch.nn.Dropout(0.2),
                                          torch.nn.Linear(256,emb_size))
        self.res = torch.nn.Sequential(torch.nn.Linear(256,256), 
                                          torch.nn.GELU(), 
                                          torch.nn.Dropout(0.2), 
                                          torch.nn.Linear(256,emb_size))
    def forward(self,x,traj):
        bs = x.shape[0]
        x = torch.cat([self.cls_token.expand(bs, -1, -1), x], dim=1)
        x = self.linear_proj_1(x)
        traj = self.traj_proj(traj)
        x = torch.cat([x,traj],dim=1)
        x = self.linear_proj_2(x)
        x = self.transformer_encoder(x)
        x = self.proj_out(x)
        latent = self.latent(x).mean(1)
        res = self.res(x).mean(1)
        return latent,res

if __name__ == '__main__':
    LATENT_SIZE = 2048
    sample_latent_net = SampleLatentNet(LATENT_SIZE).cuda()           
    print(sample_latent_net(torch.randn(128,128,2048).cuda()).shape)
