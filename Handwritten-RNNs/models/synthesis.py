import pdb
import torch
import torch.nn as nn
from utils.layers import *
from utils import ce_sample, bivariate_sample

device = check_cuda()

class Synthesis(nn.Module):
    
    def __init__(self,enc_sz=30,dec_sz=30,att_hd=20,n_g=20):
        super().__init__()
        self.char_embed =  embedding(80,enc_sz,padding_idx=0)
        self.inlayer = InLayer(fc_size=dec_sz)
        self.recur1 = RecurLayer(enc_sz+dec_sz,1)
        self.recur2 = RecurLayer(2*(enc_sz+dec_sz),1)
        self.attention = Attention(enc_sz,dec_sz,att_hd)
        self.fc = nn.Sequential(nn.Linear(2*(enc_sz+dec_sz), 3*(enc_sz+dec_sz)),
                                nn.ReLU())
        self.outlayer = MixtureDensity(3*(enc_sz+dec_sz),n_g)
        self.activation = FinalActivation()

    def forward(self,x,y,state,context,kappa):

        if state is None:
            state = [None,None]
        preds = []
        seq_len = y.shape[1]
        mask = (x != 0).unsqueeze(-1)

        enc = self.char_embed(x)
        for i in range(seq_len):
            inp = y[:,i].unsqueeze(1)
            inp = self.inlayer(inp)
            dec,state1 = self.recur1(torch.cat([inp,context],2),state[0])
            context,kappa,score = self.attention(enc,dec,kappa,mask)
            out,state2 = self.recur2(torch.cat([inp,context,dec],2),state[1])
            final = self.fc(out)
            final = self.activation(*self.outlayer(final))
            preds.append(final)
            state = [state1,state2]
        
        ce,pi,rho,sigma,mu = zip(*preds)
        return (torch.cat(ce,1),torch.cat(pi,1),torch.cat(rho,1),
                torch.cat(sigma,1),torch.cat(mu,1)), state, context, kappa

    def generate(self,x,seq_len,bias=0,temp=1):

        with torch.no_grad():
            preds, state, kappa = [], [None,None], 0
            inp = torch.as_tensor([[[2,0,0]]]).float().expand(x.size(0),1,3).to(device)
            mask = (x != 0).unsqueeze(-1)
            enc = self.char_embed(x)
            context = torch.zeros((x.size(0),1,enc.size(2)),dtype=torch.float32).to(device)
            for i in range(seq_len):
                inp = self.inlayer(inp)
                dec,state1 = self.recur1(torch.cat([inp,context],2),state[0])
                context,kappa,score = self.attention(enc,dec,kappa,mask)
                out,state2 = self.recur2(torch.cat([inp,context,dec],2),state[1])
                final = self.fc(out)
                final = self.activation(*self.outlayer(final),bias)
                state = [state1,state2]
                inp = self.gen_sample(final,temp)
                preds.append(inp.cpu())

        return torch.cat(preds,1),score


    def gen_sample(self,params,temp):

        with torch.no_grad():
            ordinal = ce_sample(params[0],temp)
            real = bivariate_sample(*params[1:])
            x = torch.cat([ordinal,real],1).unsqueeze_(1)
        return x
