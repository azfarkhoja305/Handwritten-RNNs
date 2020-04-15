import torch
import torch.nn as nn
import pdb
from utils import check_cuda, ce_sample, bivariate_sample
from utils import layers

device = check_cuda()


class AutoGen(nn.Module):
    def __init__(self, embed_dims=5, fc_size=30, rnn_layers=2,interm = 2,n_g=10):
        super().__init__()
        self.embed_dims, self.fc_size, self.rnn_layers = embed_dims, fc_size, rnn_layers
        self.inlayer = layers.InLayer(embed_dims, fc_size )
        self.recurlayer = layers.RecurLayer(fc_size,rnn_layers)
        self.inter_size = rnn_layers*fc_size
        self.fc = nn.Sequential(
                        nn.Linear(self.inter_size,interm*self.inter_size),
                        nn.ReLU())
        self.outlayer = layers.MixtureDensity(interm*self.inter_size,n_g)
        self.activation = layers.FinalActivation()

    def forward(self, x, prev_state=None,bias=0):
        x = self.inlayer(x)
        x,new_state = self.recurlayer(x,prev_state)
        x = self.fc(x)
        params = self.outlayer(x)
        params = self.activation(*params,bias)
        return params, tuple(new_state)

    def init_hidden(self):
        return tuple([None]*self.rnn_layers)

    def gen_seq(self,req_len,batch_sz=10,bias=0,temp=1):
        x = torch.as_tensor([[[2,0,0]]]*batch_sz).float().to(device)
        preds, state = [], None
        with torch.no_grad():
            for i in range(req_len):
                params,state = self.forward(x,state,bias)
                ordinal = ce_sample(params[0],temp)
                real = bivariate_sample(*params[1:])
                x = torch.cat([ordinal,real],1).unsqueeze_(1)
                preds.append(x.clone().cpu())
        return torch.cat(preds,dim=1)
        

