#!/usr/bin/env python
# coding: utf-8

# ## Train Othello-GPT and save to `ckpts`
# 
# Use `jupyter nbconvert --execute --to notebook --allow-errors --ExecutePreprocessor.timeout=-1 train_gpt_othello.ipynb --inplace --output ckpts/checkpoint.ipynb` to run in background

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# make deterministic
from mingpt.utils import set_seed
set_seed(44)


# In[3]:


import os
import math
import time
from tqdm import tqdm
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
from torch.nn import functional as F
from data import get_othello
from data.othello import permit, start_hands, OthelloBoardState, permit_reverse, unite_boards
from mingpt.dataset import CharDataset
from mingpt.utils import sample
from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig


# In[4]:


synthetic_or_championship = True  # True for training on the synthetic dataset


othello = get_othello(ood_num=-1, data_root=None if synthetic_or_championship else "data/othello_championship", wthor=True)
train_dataset = CharDataset(othello)
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, n_layer=8, n_head=8, n_embd=512)
model = GPT(mconf)


# ## Or load trained model from `ckpts`

# In[ ]:


load_res = model.load_state_dict(torch.load("./ckpts/gpt_synthetic.ckpt" if synthetic_or_championship else "./ckpts/gpt_championship.ckpt"))
if torch.cuda.is_available():
    device = torch.cuda.current_device()
    model = model.to(device)


# ## Validate it: for what percentage of all partial games in validation set, the top-1 prediction is legal

# In[ ]:


if not synthetic_or_championship:  # for GPT trained on both datasets, use the validation set of synthetic for validation
    othello = get_othello(ood_num=-1, data_root=None, wthor=True)


# In[ ]:


total_nodes = 0
success_nodes = 0

bar = tqdm(othello.val[:1000])
for whole_game in bar:
    length_of_whole_game = len(whole_game)
    for length_of_partial_game in range(1, length_of_whole_game):
        total_nodes += 1
        context = whole_game[:length_of_partial_game]
        x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None, ...].to(device)
        y = sample(model, x, 1, temperature=1.0)[0]
        completion = [train_dataset.itos[int(i)] for i in y if i != -1]
        try:
            OthelloBoardState().update(unite_boards(completion), prt=False)
        except Exception:
#             fail_nodes.append([permit_reverse(_) for _ in context])
            pass
        else:
            success_nodes += 1
    bar.set_description(f"{success_nodes/total_nodes*100:.2f}% pass rate: {success_nodes}/{total_nodes} among all searched nodes")
print(f"{success_nodes/total_nodes*100:.2f}% pass rate: {success_nodes}/{total_nodes} among all searched nodes")


# In[ ]:


1 - success_nodes/total_nodes

