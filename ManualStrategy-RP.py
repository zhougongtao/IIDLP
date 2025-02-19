import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

import os
import json
import time
import argparse

from EIG import *
from Model import *
from data.InvestorContext import *

def parse_args():
    parser = argparse.ArgumentParser(description='Manual Strategy Neural Network')
    # manual strategy
    parser.add_argument('--strategy', type=int, default=0, help='differenet manual strategy')
    # train
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Adam: weight decay - L2 ')
    parser.add_argument('--T', type=int, default=100, help='number of period')
    parser.add_argument('--batch', type=int, default=50, help='number of batches')
    # early stopping
    parser.add_argument('--es_delta', type=float, default=0, help='delta of early stopping')
    parser.add_argument('--max_NN_epoch', type=int, default=5000, help='max number of NN epoch')
    parser.add_argument('--NN_patience', type=int, default=10, help='patience of NN early stopping')
    # data    
    parser.add_argument('--data', type=str, default='dataA', help='name of data')
    parser.add_argument('--noise', type=float, default=0.1, help='noise of data')
    # design
    parser.add_argument('--design_num', type=int, default=10, help='number of designs')
    parser.add_argument('--portfolio_num', type=int, default=2, help='number of portfolios')
    # other
    parser.add_argument('--seed', type=int, default=-1, help='seed')
    parser.add_argument('--print', action='store_true', help='print some detial')
    parser.add_argument('--dataset', action='store_true', help='generate dataset during train')
        
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
    else:
        print("error manual strategy")
        exit()
    
    args.name = f"manual_strategy2_RP/{args.data}-T{args.T}-batch{args.batch}-design{args.design_num}-{args.portfolio_num}-noise{args.noise}-{args.strategy_name}"
    # seed 
    if args.seed == -1:
        args.seed = int.from_bytes(os.urandom(4), 'big')
    torch.manual_seed(args.seed)
    return args

def train(train_loader, test_loader, online_dataset, model, optimizer, args):
    iter_num = 0
    for epoch, (x, lam, risk, design, select) in enumerate(train_loader):
        
        # print(f"---------- epoch: {epoch} ----------")
        # get design
        design_idx = None
        if epoch == 0: 
            design_idx = 0
        else:
            if args.strategy == 0:
                design_idx = torch.randint(0, args.design_num, ())
            elif args.strategy in [1, 2, 3, 4]:
                mean = design[0][:, :, 0].sum(axis=1)
                std = design[0][:, :, 1].sum(axis=1)
                if args.strategy == 1: arr = mean
                if args.strategy == 2: arr = std
                if args.strategy == 3: arr = mean + std
                if args.strategy == 4: arr = mean - std
                design_idx = torch.argmax(arr)
                
            elif args.strategy in [5, 6]:
                runs_times = 20 
                sum_best_val_loss = float('inf')
                for times in range(runs_times):

                    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
                
                    online_train_loader, online_val_loader = getOnlineDataLoader(online_dataset, batch_size=args.batch)
                    early_stopping = EarlyStopping(args.NN_patience, args.es_delta)

                    # Initialize variables
                    best_val_loss = float('inf')  # Initialize with a very large number
                    model.load_state_dict(torch.load(f'model_weights/NN_initial.pth'))

                    # train
                    NN_start_time = time.time()
                    for NN_epoch in range(args.max_NN_epoch):
                        model.train()
                        total_num = 0
                        total_nll_loss, total_lam_mse, total_risk_mse = 0., 0., 0.
                        for _x, _lam, _risk, _d, _, _select in online_train_loader:
                            model.zero_grad()
                            pred_lam, pred_risk = model(_x)
                            log_prob = prob_portfolio(pred_lam, pred_risk, _d)

                            nll_loss = F.nll_loss(log_prob, _select)
                            lam_mse = F.mse_loss(pred_lam.squeeze(), _lam)
                            risk_mse = F.mse_loss(pred_risk.squeeze(), _risk)

                            nll_loss.backward()
                            optimizer.step()

                            total_num += _x.size(0)
                            total_nll_loss += nll_loss * _x.size(0)
                            total_lam_mse += lam_mse * _x.size(0)
                            total_risk_mse += risk_mse * _x.size(0)

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

                        # Perform early stopping check and model saving
                        if total_nll_loss < best_val_loss:
                            best_val_loss = total_nll_loss
                            # print(best_val_loss/val_total_num)
                            torch.save(model.state_dict(), f'model_weights/{args.strategy_name}_{args.data}_{args.noise}_{args.seed}_best_duringtraining.pth')
                            if best_val_loss < sum_best_val_loss:
                                torch.save(model.state_dict(), f'model_weights/{args.strategy_name}_{args.data}_{args.noise}_{args.seed}_sum_best_duringtraining.pth')
                                sum_best_val_loss = best_val_loss
                        
                        early_stopping(total_nll_loss)
                        if early_stopping.early_stop or NN_epoch + 1 == args.max_NN_epoch:
                            if args.print: print('earlystop')
                            if args.print: print(NN_epoch)
                            # print(total_nll_loss/val_total_num)
                            break
        
                model.load_state_dict(torch.load(f'model_weights/{args.strategy_name}_{args.data}_{args.noise}_{args.seed}_sum_best_duringtraining.pth'))
                model.eval()
                with torch.no_grad():
                    metrics = torch.zeros(args.design_num)
                    pred_lam, pred_risk = model(x)
                    for d_i in range(args.design_num):
                        log_prob = prob_portfolio(pred_lam, pred_risk, design[0][d_i])  
                        prob = torch.exp(log_prob)
                        
                        if args.strategy == 5: 
                            metrics[d_i] = (prob * log_prob).sum()
                        if args.strategy == 6: 
                            max_values, _ = torch.max(log_prob, dim=1)
                            metrics[d_i] = max_values.sum()
                    design_idx = torch.argmin(metrics)
                    
        # add online dataset
        epoch_list = torch.full((args.batch,), epoch)
        online_dataset.extend(x, lam, risk, design[:,design_idx], epoch_list, select[:,design_idx])
        
        # dataset
        if args.dataset and epoch + 1 in range(1, 16):
            dir = f"dataset/{args.seed}/{args.strategy_name}/"
            file = f"{args.data}-{args.noise}-epoch{epoch+1}"
            save_dataset(online_dataset, dir, file)
        if epoch + 1 == 15: return

def save_dataset(dataset, directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)

    filepath = os.path.join(directory, filename + '.pt')
    torch.save(dataset, filepath)
    print(f"Dataset saved to {filepath}")

def main(): 
    args = parse_args()
    if args.print: print(args)
    
    train_loader, test_loader = get_dataloader(args.data, args.T, args.batch, args.noise)
    online_dataset = InvestorOnlineDataSet()
    
    model = SimpleNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    train_time = time.time()
    train(train_loader, test_loader, online_dataset, model, optimizer, args)
    print(f"Done! train time: {time.time() - train_time}")

if __name__ == '__main__':
    main()
