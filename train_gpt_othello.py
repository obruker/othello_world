#!/usr/bin/env python
# coding: utf-8

# ## Train Othello-GPT and save to `ckpts`
# 
# Use `jupyter nbconvert --execute --to notebook --allow-errors --ExecutePreprocessor.timeout=-1 train_gpt_othello.ipynb --inplace --output ckpts/checkpoint.ipynb` to run in background

# In[1]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


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
from data.othello import permit, start_hands, OthelloBoardState, permit_reverse
from mingpt.dataset import CharDataset
from mingpt.utils import sample
from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig


# In[4]:


synthetic_or_championship = True  # True for training on the synthetic dataset


# In[5]:


othello = get_othello(ood_num=-1, data_root=None if synthetic_or_championship else "data/othello_championship", wthor=True)
train_dataset = CharDataset(othello)
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, n_layer=8, n_head=8, n_embd=512)
model = GPT(mconf)


# In[ ]:


max_epochs = 3
# initialize a trainer instance and kick off training
t_start = time.strftime("_%Y%m%d_%H%M%S")
tconf = TrainerConfig(
    max_epochs=max_epochs, 
    batch_size=512,  # reduced from 512*8 due to memory limitations
    learning_rate=5e-4,
    lr_decay=True, 
    warmup_tokens=len(train_dataset)*train_dataset.block_size*5, 
    final_tokens=len(train_dataset)*train_dataset.block_size*max_epochs,
    num_workers=0, 
    ckpt_path=f"./ckpts/gpt_at{t_start}.ckpt", 
)
trainer = Trainer(model, train_dataset, None, tconf)
device = trainer.device
print(t_start)
trainer.train()
