import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from sklearn.model_selection import KFold
import numpy as np

import os
import json
import time
import argparse

from EIG import *
from Model import *
from data.InvestorContext import *

def parse_args():
    parser = argparse.ArgumentParser(description='test dataset by NN')
    # train
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Adam: weight decay - L2 ')
    parser.add_argument('--batch', type=int, default=50, help='number of batches')
    # early stopping
    parser.add_argument('--max_NN_epoch', type=int, default=5000, help='max number of NN epoch')
    parser.add_argument('--NN_patience', type=int, default=10, help='patience of NN early stopping')
    # data    
    # parser.add_argument('--data', type=str, default='data1', help='name of data')
    parser.add_argument('--noise', type=float, default=0.1, help='noise of data')
    # design
    parser.add_argument('--design_num', type=int, default=10, help='number of designs')
    parser.add_argument('--portfolio_num', type=int, default=2, help='number of portfolios')
    # other
    parser.add_argument('--strategy', type=int, default=0, help='differenet manual strategy')
    
    args = parser.parse_args() 
    # writer
    if args.strategy == 0:
        args.strategy_name = "Random"
    elif args.strategy == 1:
        args.strategy_name = "MaxMean"
    elif args.strategy == 2:
        args.strategy_name = "MaxVar"
    elif args.strategy == 3:
        args.strategy_name = "MaxMean+Var"
    elif args.strategy == 4:
        args.strategy_name = "MaxMean-Var"
    elif args.strategy == 5:
        args.strategy_name = "PreEntropy"
    elif args.strategy == 6:
        args.strategy_name = "MinMaxPro"
    elif args.strategy == 11:
        args.strategy_name = "IIDLP"
    else:
        print("error manual strategy")
        exit()
        
    return args

def train(online_dataset, test_loader, args): 
    model = SimpleNN()
    
    K = 5
    kfold = KFold(n_splits=K, shuffle=True, random_state=42)
    final_nll_loss, final_lam_mse, final_risk_mse = 0., 0., 0.
    for fold, (train_index, val_index) in enumerate(kfold.split(online_dataset)):
        train_subset = Subset(online_dataset, train_index)
        val_subset = Subset(online_dataset, val_index)
        
        online_train_loader = DataLoader(train_subset, batch_size=args.batch, shuffle=True)
        online_val_loader = DataLoader(val_subset, batch_size=args.batch, shuffle=False)
        
        # model
        model.load_state_dict(torch.load(f'model_weights/NN_initial.pth'))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        best_val_loss = float('inf')
        early_stopping_NN = EarlyStopping(patience=args.NN_patience)
        
        # Offline train NN 
        for NN_epoch in range(args.max_NN_epoch):
            model.train()
            for _x, _lam, _risk, _d, _, _select in online_train_loader:      
                model.zero_grad()
                pred_lam, pred_risk = model(_x)
                log_prob = prob_portfolio(pred_lam, pred_risk, _d)
                
                nll_loss = F.nll_loss(log_prob, _select)
                nll_loss.backward()
                optimizer.step()
        
            model.eval()
            val_total_num = 0
            total_nll_loss = 0.
            with torch.no_grad():
                for _x, _lam, _risk, _d, _, _select in online_val_loader:  
                    pred_lam, pred_risk = model(_x)
                    log_prob = prob_portfolio(pred_lam, pred_risk, _d)  

                    nll_loss = F.nll_loss(log_prob, _select)
                    total_nll_loss += nll_loss
                    val_total_num += _x.shape[0]

            if total_nll_loss < best_val_loss:
                best_val_loss = total_nll_loss
                torch.save(model.state_dict(), f'model_weights/{args.strategy_name}_best_duringtraining.pth')
            
            early_stopping_NN(total_nll_loss)
            if early_stopping_NN.early_stop or NN_epoch + 1 == args.max_NN_epoch:
                model.load_state_dict(torch.load(f'model_weights/{args.strategy_name}_best_duringtraining.pth'))
                break
    
        model.eval()
        val_total_num = 0
        total_nll_loss, total_lam_mse, total_risk_mse = 0., 0., 0. 
        with torch.no_grad():
            for x, lam, risk, d, select in test_loader:  
                pred_lam, pred_risk = model(x)
                log_prob = prob_portfolio(pred_lam, pred_risk, d)  

                nll_loss = F.nll_loss(log_prob, select)
                lam_mse = F.mse_loss(pred_lam.squeeze(), lam)
                risk_mse = F.mse_loss(pred_risk.squeeze(), risk)
                
                val_total_num += x.shape[0]
                total_nll_loss += nll_loss
                total_lam_mse += lam_mse
                total_risk_mse += risk_mse
                
        final_nll_loss += total_nll_loss / val_total_num
        final_lam_mse += total_lam_mse / val_total_num
        final_risk_mse += total_risk_mse / val_total_num
       
    final_nll_loss = final_nll_loss / K * args.batch
    final_lam_mse = final_lam_mse / K * args.batch
    final_risk_mse = final_risk_mse / K * args.batch
    
    return final_nll_loss, final_lam_mse, final_risk_mse
    
def run(args):
    seeds = [
        2190830449, 3538903541, 1967909051, 1156027684, 2702614907,
        2131527277, 7466345345, 2131516612, 7456363156, 9576586795,
        7252349820, 6284327691, 5624752851, 6384519612, 1571681920,
        2131511616, 1516172371, 4634763472, 1515162735, 6516324354,
    ] 
    res = {}
    for setting in ["A","B","C","D","E"]:
        data = f"data{setting}"
        for noise in [0.1]:
            _, test_loader = get_dataloader(data, 100, 50, noise)  
            res[f"{data}-{noise}"] = {}
            for epoch in range(1, 16):
                nll_loss_list = []
                lam_mse_list = []
                risk_mse_list = []
                for seed in seeds:
                    torch.manual_seed(seed)
                    dir = f"dataset/{seed}/{args.strategy_name}/"
                    file = f"{data}-{noise}-epoch{epoch}.pt"
                    online_dataset = torch.load(dir + file)
                    
                    nll_loss, lam_mse, risk_mse = train(online_dataset, test_loader, args)
                    nll_loss_list.append(nll_loss.item())
                    lam_mse_list.append(lam_mse.item())
                    risk_mse_list.append(risk_mse.item())
                    
                res[f"{data}-{noise}"][f'epoch-{epoch}'] = {"nll_loss":nll_loss_list, "lam_mse":lam_mse_list, "risk_mse":risk_mse_list}
            print(f"{args.strategy_name}-{data}-noise{noise} done!!!")
            
    if not os.path.exists("./result/"): os.makedirs("./result/")
    file_name = f"./result/{args.strategy_name}.json"
    out_file = open(file_name, "w") 
    json.dump(res, out_file, indent = 4) 
    out_file.close()    
    print(f"{time.asctime()}   {file_name} done!!!")
    
if __name__ == '__main__':
    args = parse_args()
    model = SimpleNN()
    run(args)