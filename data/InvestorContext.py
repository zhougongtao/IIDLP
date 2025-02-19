import os
import json
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

"""
    DataSet Class:
        InvestorTrainDataSet: train dataset (design (10, 2, 2))
        InvestorTestDataSet: test dataset   (portfolios (2, 2))
        InvestorOnlineDataSet: collect online dataset
"""
class InvestorTrainDataSet(Dataset):
    def __init__(self, X, lam, risk, design, select):
        self.X = X
        self.lam = lam
        self.risk = risk
        self.design = design
        self.select = select
 
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.lam[idx], self.risk[idx], self.design[idx], self.select[idx]
    
class InvestorTestDataSet(Dataset):
    def __init__(self, X, lam, risk, portfolio, select):
        self.X = X
        self.lam = lam
        self.risk = risk
        self.portfolio = portfolio
        self.select = select

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.lam[idx], self.risk[idx], self.portfolio[idx], self.select[idx]

def get_dataloader(data, T, batch, noise):
    train_dataset = torch.load(f"./data/InvestorContext/{data}_train_T{T}_batch{batch}_noise{noise}.pt")
    test_dataset = torch.load(f"./data/InvestorContext/{data}_test_T{T}_batch{batch}_noise{noise}.pt")

    train_loader = DataLoader(train_dataset, batch_size=batch)
    test_loader = DataLoader(test_dataset, batch_size=batch)
    return train_loader, test_loader

class InvestorOnlineDataSet(Dataset):
    def __init__(self):
        self.X = torch.empty(0, 16)
        self.lam = torch.empty(0)
        self.risk = torch.empty(0)
        self.portfolio = torch.empty(0,2,2)
        self.epoch = torch.empty(0, dtype=torch.int64)
        self.select = torch.empty(0, dtype=torch.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.lam[idx], self.risk[idx], self.portfolio[idx], self.epoch[idx], self.select[idx]
    
    def extend(self, X, lam, risk, portfolio, epoch, select):
        self.X = torch.cat((self.X, X))
        self.lam = torch.cat((self.lam, lam))
        self.risk = torch.cat((self.risk, risk))
        self.portfolio = torch.cat((self.portfolio, portfolio))
        self.epoch = torch.cat((self.epoch, epoch))
        self.select = torch.cat((self.select, select))
        
        # shuffle
        shuffle_indices = torch.randperm(self.X.size(0))
        self.X = self.X[shuffle_indices]
        self.lam = self.lam[shuffle_indices]
        self.risk = self.risk[shuffle_indices]
        self.portfolio= self.portfolio[shuffle_indices]
        self.epoch = self.epoch[shuffle_indices]
        self.select = self.select[shuffle_indices]        
        
def getOnlineDataLoader(online_dataset, batch_size=50, ratio=0.8):
    train_size = int(ratio * len(online_dataset))
    test_size = len(online_dataset) - train_size

    train_dataset, val_dataset = random_split(online_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def prob_portfolio(lam, risk, portfolio):
    """
        lam and risk : [..., 1]
        portfolios : [N_PORTFPLIO, 2] or [batch, N_PORTFPLIO, 2]
    """
    N_PORTFPLIO = portfolio.shape[-2]
    m = portfolio[...,0].view(-1, N_PORTFPLIO) 
    v = portfolio[...,1].view(-1, N_PORTFPLIO) 
    revenue = lam * (m - risk * v)        # [..., N_PORTFPLIO]
    log_prob = F.log_softmax(revenue, dim=-1)
    return log_prob  

