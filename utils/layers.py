import torch
import torch.nn as nn
import pdb
from utils import check_cuda

device = check_cuda()


## Credit https://github.com/fastai/fastai/blob/master/fastai/layers.py#L285

def trunc_normal_(x, mean:float=0., std:float=1.):
    "Truncated normal initialization."
    return x.normal_().fmod_(2).mul_(std).add_(mean)

def embedding(ni,nf,padding_idx=None):
    "Create an embedding layer."
    emb = nn.Embedding(ni, nf, padding_idx)
    # See https://arxiv.org/abs/1711.09160
    with torch.no_grad(): trunc_normal_(emb.weight, std=0.01)
    return emb


class RecurLayer(nn.Module):
    """Multiple LSTM Layers with skip input connections.
        Gives all layer outsputs and states"""
    def __init__(self,dims=10,num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.rnns = nn.ModuleList([nn.LSTM(dims,dims,batch_first=True)])
        for i in range(num_layers-1): 
            self.rnns.append(nn.LSTM(dims*2,dims,batch_first=True))
    def forward(self,x,prev_state=None):
        if prev_state is None:
            prev_state = [None]*self.num_layers
        cat_xs,new_state = [],[]
        skip_x  = x.clone() 
        for i in range(self.num_layers):
            new_x,s = self.rnns[i](x,prev_state[i])
            cat_xs.append(new_x)  
            new_state.append(s) 
            x = torch.cat([skip_x,new_x],2) # skip connection inputs
        return torch.cat(cat_xs,2), new_state  # skip connection hidden


class InLayer(nn.Module):
    def __init__(self,embed_dims=5, fc_size = 20):
        super().__init__()
        # 0,1,start,end,pad
        self.embed = embedding(5,embed_dims, padding_idx=4)
        self.fc1 = nn.Linear(2,embed_dims)
        self.fc2 = nn.Linear(2*embed_dims,fc_size)
        self.activation = nn.Tanh()
    def forward(self,x):
        ordinal, real = x[:,:,0].long(), x[:,:,1:]
        ordinal = self.embed(ordinal)
        real = self.fc1(real)
        return self.activation(self.fc2(torch.cat([ordinal,real],2)))



# Inspiration from https://github.com/sagelywizard/pytorch-mdn/blob/master/mdn.py

class MixtureDensity(nn.Module):
    def __init__(self,ip_dims,num):
        super().__init__()
        self.num = num
        self.ce = nn.Linear(ip_dims,5)
        self.pi = nn.Linear(ip_dims,num)
        self.sigma = nn.Linear(ip_dims,num*2)
        self.rho = nn.Linear(ip_dims,num)
        self.mu = nn.Linear(ip_dims,num*2)
        
    def forward(self,x):
        ce = self.ce(x)
        pi = self.pi(x)
        rho = self.rho(x)
        shape = x.size()
        sigma = self.sigma(x).view(shape[0],shape[1],self.num,2)
        mu = self.mu(x).view(shape[0],shape[1],self.num,2)

        return ce,pi,rho,sigma,mu


class FinalActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()
        self.tanh = nn.Tanh()
    def forward(self,ce,pi,rho,sigma,mu,bias=0):
        pi = self.softmax(pi*(1+bias))
        rho = self.tanh(rho)
        sigma = self.softplus(sigma-bias)
        return ce,pi,rho,sigma,mu

class Encoder(nn.Module):
    def __init__(self,num_em=80,embed_sz=30,rnn_layers=1):
        super().__init__()
        self.embed = nn.Embedding(num_em,embed_sz,padding_idx=0)
        self.rnns = RecurLayer(embed_sz,rnn_layers)
    def forward(self,x):
        x = self.embed(x)
        x,_ = self.rnns(x)
        return x


class Decoder(nn.Module):
    def __init__(self,embed_dims=5,fc_size=30,rnn_layers=2,enc_sz=30):
        super().__init__()
        self.inlayer = InLayer(embed_dims, fc_size)
        self.recurlayer = RecurLayer(fc_size + enc_sz,rnn_layers)
    def forward(self,x,context, prev_state=None):
        x = self.inlayer(x)
        x = torch.cat([x,context],2)
        x,new_state = self.recurlayer(x,prev_state)
        return x, new_state



class Attention(nn.Module):
    
    def __init__(self,enc_sz,dec_sz,num=10):
        super().__init__()

        self.alpha = nn.Sequential(nn.Linear(dec_sz+enc_sz,num),
                                  nn.Softplus())
        self.beta = nn.Sequential(nn.Linear(dec_sz+enc_sz,num),
                                  nn.Softplus())
        self.new_kappa =  nn.Sequential(nn.Linear(dec_sz+enc_sz,num),
                                  nn.Softplus())

    def forward(self,enc,dec,kappa,mask):
        alpha = self.alpha(dec)
        beta = self.beta(dec)
        kappa = kappa + self.new_kappa(dec)
        u = torch.arange(enc.size(1)).float().view(1,-1,1).cuda()
        score = alpha*(torch.exp(-beta*(kappa-u)**2))
        score = score.sum(-1, keepdim=True)
        context = (enc*mask*score).sum(1, keepdim=True)
        return context, kappa, score



class SeqAttention(nn.Module):
    
    def __init__(self,enc_sz,dec_sz,heads,interm=30):
        super().__init__()
        self.heads = heads
        self.value = nn.Linear(enc_sz,interm*heads)
        self.query = nn.Linear(dec_sz+enc_sz,interm*heads)
        self.final = nn.Linear(interm*heads,heads)
        self.softplus = nn.Softplus()
        self.activation = nn.Tanh()
        self.beta = nn.Sequential(nn.Linear(dec_sz+enc_sz,heads),
                                  nn.Softplus())
        self.new_kappa =  nn.Sequential(nn.Linear(dec_sz+enc_sz,heads),
                                  nn.Softplus())

        self.fc = nn.Linear(enc_sz*heads,enc_sz)

    def forward(self,enc,dec,kappa,mask):
        bs,seq_len = enc.shape[0],enc.shape[1]
        value = self.value(enc)
        query = self.query(dec).expand_as(value)
        alpha = self.softplus(self.final(self.activation(query+value)))
        alpha = alpha * mask
        beta = self.beta(dec).view(bs,1,self.heads).expand_as(alpha)
        kappa = kappa + self.new_kappa(dec).view(bs,1,self.heads).expand_as(alpha)
        u = torch.arange(enc.size(1),dtype=torch.float32).view(1,-1,1).to(device)
        score = alpha*(torch.exp(-beta*(kappa-u)**2))
        context = (enc.unsqueeze(-2)*score.unsqueeze(-1)).sum(1).view(bs,-1)
        context = self.fc(context)
        return context.unsqueeze(1), kappa, score









