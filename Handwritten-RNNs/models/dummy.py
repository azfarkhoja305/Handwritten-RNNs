import numpy
import torch
import copy

from models.synthesis import Synthesis
from models.autogen import AutoGen
from utils import *
from utils.other_utils import *

device = check_cuda()

strokes = numpy.load('./gdrive/My Drive/descript-research-test/data/strokes-py3.npy', allow_pickle=True)
process_strokes = ProcessStrokes()
norm_strokes = process_strokes.normalize(copy.deepcopy(strokes))
final_strokes =  process_strokes.add_tokens(copy.deepcopy(norm_strokes))


with open('./gdrive/My Drive/descript-research-test/data/sentences.txt') as f:
    texts = f.readlines()
process_text = ProcessText()
char_list = process_text.tokenize(texts)
word2idx,idx2word = process_text.create_vocab()
char_data = process_text.numericalize()

model1 = AutoGen(fc_size=30, interm=4 ,rnn_layers=3, n_g=20)
model1.load_state_dict(torch.load('./gdrive/My Drive/descript-research-test/unconditional.pt', map_location=device))
model1.to(device);

model2 = Synthesis(enc_sz=70,dec_sz = 30,att_hd = 20,n_g = 20)
model2.load_state_dict(torch.load('./gdrive/My Drive/descript-research-test/conditional.pt', map_location=device))
model2.to(device);


def generate_unconditionally(random_seed=1):
    # Input:
    #   random_seed - integer
    gen_strokes = model1.gen_seq(700,bias=0.15,temp=0.75)
    one_stroke = process_strokes.denormalize(gen_strokes[0].cpu(),True)
    # Output:
    #   stroke - numpy 2D-array (T x 3)
    return one_stroke


def generate_conditionally(text='welcome to lyrebird', random_seed=1):
    # Input:
    #   text - str
    #   random_seed - integer
    x1 = torch.as_tensor(process_text.sent2num(text))[None]
    x2 = x1.new_zeros(x1.shape)
    x = torch.cat([x1,x2],0).to(device)
    ans,_ = model2.generate(x,1800,bias=1.05,temp=1)
    one_stroke = process_strokes.denormalize(ans[0].cpu(),True)
    # Output:
    #   stroke - numpy 2D-array (T x 3)
    return one_stroke


def recognize_stroke(stroke):
    # Input:
    #   stroke - numpy 2D-array (T x 3)

    # Output:
    #   text - str
    return 'welcome to lyrebird'