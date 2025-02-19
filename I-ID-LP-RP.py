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
    parser = argparse.ArgumentParser(description='Integrated Inference-and-Design with Learnable Priors')
    parser.add_argument('--strategy', type=int, default=11, help='differenet manual strategy')
    # train
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Adam: weight decay - L2 ')
    parser.add_argument('--T', type=int, default=100, help='number of period')
    parser.add_argument('--batch', type=int, default=50, help='number of batches')
    parser.add_argument('--sample', type=int, default=50, help='number of sample model weight during train')
    # early stopping
    parser.add_argument('--es_delta', type=float, default=0, help='delta of early stopping')
    parser.add_argument('--max_NN_epoch', type=int, default=5000, help='max number of NN epoch')
    parser.add_argument('--NN_patience', type=int, default=10, help='patience of NN early stopping')
    parser.add_argument('--max_BNN_epoch', type=int, default=200, help='max number of BNN epoch')
    parser.add_argument('--BNN_patience', type=int, default=10, help='patience of BNN early stopping')
    parser.add_argument('--max_design_epoch', type=int, default=10, help='max number of design epoch')
    parser.add_argument('--design_patience', type=int, default=2, help='patience of design early stopping')
    # MOPED
    parser.add_argument('--prior_var', type=float, default=100, help='MOPED model')
    parser.add_argument('--delta', type=float, default=0.5, help='MOPED model')
    # EIG
    parser.add_argument('--NMC_L', type=int, default=1500, help='sample for NMC')
    parser.add_argument('--NMC_K_train_fast', type=int, default=1500, help='fast sample of NMC')
    parser.add_argument('--NMC_K_fast', type=int, default=1500, help='fast sample of NMC')
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
    args.name = f"I-ID-LP/{args.data}-T{args.T}-batch{args.batch}-noise{args.noise}-design{args.design_num}-{args.portfolio_num}"
    # seed
    if args.seed == -1:
        args.seed = int.from_bytes(os.urandom(4), 'big')
    torch.manual_seed(args.seed)
    return args


def train(train_loader, test_loader, online_dataset, model, optimizer, args):
    iter_num = 0

    for epoch, (x, lam, risk, design, select) in enumerate(train_loader):
        if args.print: print(f"---------- epoch: {epoch} ----------")
        
        design_idx = torch.randint(args.design_num, ())

        if epoch == 0:
            design_idx = 0

        if epoch > 0:
            # MOPED method
            online_all_loader = DataLoader(online_dataset, args.batch, shuffle=True)  #  * epoch
            model.load_state_dict(torch.load(f'model_weights/I_IDLP_{args.data}_{args.noise}_{args.seed}_sum_best_duringtraining.pth'))
            BNN = BayesianNetworkWithMOPED(model, args.delta, args.prior_var)
            optimizer_BNN = torch.optim.Adam(BNN.parameters(), lr=args.lr*100)
            
            design_cnt = 0
            design_pre_cnt = torch.zeros(args.design_num)

            best_design_idx = None
            bnn_best_train_loss_across_design = float('inf')

            while design_cnt < args.design_patience:
                """
                    train phi
                    fixed design, train BNN
                """

                # Initialize variables
                bnn_best_train_loss = float('inf')  # Initialize with a very large number
                
                d = design[0][design_idx]
                BNN_start_time = time.time()
                early_stopping_BNN = EarlyStopping(args.BNN_patience, args.es_delta)
                for BNN_epoch in range(args.max_BNN_epoch):
                    BNN_train_loss = 0.
                    BNN_train_total_num = 0
                    BNN.train()
                    data_num = len(online_dataset)
                    kl_weight = 0 if data_num == 0 else (50 / data_num)

                    # '''
                    for _x, _lam, _risk, _d, _, _select in online_all_loader:    
                        BNN.zero_grad()
                        kl, log_like, lam_mse, risk_mse, logEIG = 0., 0., 0., 0., 0.
                        for _ in range(args.sample):
                            pred_lam, pred_risk = BNN(_x)
                            log_prob = prob_portfolio(pred_lam, pred_risk, _d)
                            
                            kl += BNN.kl()
                            log_like += -F.nll_loss(log_prob, _select)
                            lam_mse += F.mse_loss(pred_lam.squeeze(), _lam)
                            risk_mse += F.mse_loss(pred_risk.squeeze(), _risk)
                        if design_cnt > 0:  # in the first iteration, do the same thing as ID-LP. This is to use the ID-LP design as a good initialization to save iteration time
                            logEIG += nmc_eig_EM_fast(BNN, x, d, args)
                            
                        # average
                        kl, log_like, logEIG = kl/args.sample, log_like/args.sample, logEIG/1
                        lam_mse, risk_mse = lam_mse/args.sample, risk_mse/args.sample
                        loss = kl * kl_weight - log_like - logEIG * kl_weight * 10
                        
                        BNN_train_loss += loss
                        BNN_train_total_num += _x.shape[0]
                        
                        loss.backward()
                        optimizer_BNN.step()

                    # Perform early stopping check and model saving
                    if BNN_train_loss < bnn_best_train_loss:
                        bnn_best_train_loss = BNN_train_loss
                        torch.save(BNN.state_dict(), f'model_weights/I_IDLP_{args.data}_{args.noise}_{args.seed}_BNN_best_duringtraining.pth')
                        if args.print: print(bnn_best_train_loss / BNN_train_total_num)

                    early_stopping_BNN(BNN_train_loss)
                    if early_stopping_BNN.early_stop or BNN_epoch + 1 == args.max_BNN_epoch:
                        BNN.load_state_dict(torch.load(f'model_weights/I_IDLP_{args.data}_{args.noise}_{args.seed}_BNN_best_duringtraining.pth'))
                        BNN_end_time = time.time()
                        if args.print: print(f"BNN train {BNN_epoch + 1} epoch, time {BNN_end_time - BNN_start_time} s")
                        break

                if (bnn_best_train_loss / BNN_train_total_num) < bnn_best_train_loss_across_design and (design_cnt > 0):  # design_cnt = 0 is the special case (same as ID-LP)
                    bnn_best_train_loss_across_design = bnn_best_train_loss / BNN_train_total_num
                    best_design_idx = design_idx

                """ 
                    train design
                    fixed BNN, get optimal design
                """          
                BNN.eval()

                if design_cnt == 0:  # in the first iteration, do the same thing as ID-LP. This is to use the ID-LP design as a good initialization to save iteration time
                    cur_design_idx, _ = nmc_eig_y(BNN, x, design[0], args)
                else:
                    EIG = torch.zeros(args.design_num)
                    with torch.no_grad():
                        for d_i in range(args.design_num):
                            EIG[d_i] += nmc_eig_EM_fast_calculate(BNN, x, design[0][d_i], args)
                    cur_design_idx = torch.argmax(EIG).item()
                
                """
                    design early stop
                """
                if design_idx == cur_design_idx:
                    design_cnt += 1
                else:
                    design_idx = cur_design_idx
                    design_cnt = 1
                    
                # reach max design epoch
                design_pre_cnt[design_idx] += 1
                if torch.sum(design_pre_cnt).item() == args.max_design_epoch or \
                   design_pre_cnt[design_idx] > args.max_design_epoch // 2:
                    design_idx = cur_design_idx   
                    design_cnt = args.design_patience
                    
                if args.print: print(f"design_idx {design_idx} design_cnt {design_cnt}")
                if args.print: print(f"\tEIG max {EIG.max():.6f}; min {EIG.min():.6f}; mean {EIG.mean():.6f}; std {EIG.std():.6f}")
            
            del BNN # clear model  
                          
        # add online dataset
        if args.print: print(f"design choose {design_idx}")
        epoch_list = torch.full((args.batch,), epoch)
        online_dataset.extend(x, lam, risk, design[:,design_idx], epoch_list, select[:,design_idx])
        
        """
            train NN
        """

        runs_times = 20
        
        sum_best_val_loss = float('inf')
        
        for times in range(runs_times):

            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            online_train_loader, online_val_loader = getOnlineDataLoader(online_dataset, batch_size=args.batch)
            early_stopping_NN = EarlyStopping(args.NN_patience, args.es_delta)

            # Initialize variables
            best_val_loss = float('inf')  # Initialize with a very large number

            model.load_state_dict(torch.load(f'model_weights/NN_initial.pth'))

            NN_start_time = time.time()
            for NN_epoch in range(args.max_NN_epoch):
                model.train()
                total_num = 0
                total_nll_loss, total_lam_mse, total_risk_mse = 0., 0., 0.,
                for _x, _lam, _risk, _d, _, _select in online_train_loader:
                    model.zero_grad()
                    pred_lam, pred_risk = model(_x)
                    log_prob = prob_portfolio(pred_lam, pred_risk, _d)

                    nll_loss = F.nll_loss(log_prob, _select)
                    lam_mse = F.mse_loss(pred_lam.squeeze(), _lam)
                    risk_mse = F.mse_loss(pred_risk.squeeze(), _risk)
                    nll_loss.backward()
                    optimizer.step()

                    total_num += _x.shape[0]
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
                    torch.save(model.state_dict(), f'model_weights/I_IDLP_{args.data}_{args.noise}_{args.seed}_best_duringtraining.pth')
                    if best_val_loss < sum_best_val_loss:
                        torch.save(model.state_dict(), f'model_weights/I_IDLP_{args.data}_{args.noise}_{args.seed}_sum_best_duringtraining.pth')
                        sum_best_val_loss = best_val_loss

                early_stopping_NN(total_nll_loss)
                if early_stopping_NN.early_stop or NN_epoch + 1 == args.max_NN_epoch:

                    if args.print: print('earlystop')
                    if args.print: print(NN_epoch)
                    if args.print: print(total_nll_loss/val_total_num)
                    model.load_state_dict(torch.load(f'model_weights/I_IDLP_{args.data}_{args.noise}_{args.seed}_best_duringtraining.pth'))

                    NN_end_time = time.time()
                    if args.print: print(f"NN train {NN_epoch + 1} epoch, time {NN_end_time - NN_start_time} s")
                    break
            
        # dataset
        if args.dataset and epoch + 1 in range(1, 16):
            dir = f"dataset/{args.seed}/IIDLP/"
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
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    train_time = time.time()
    train(train_loader, test_loader, online_dataset, model, optimizer, args)
    print(f"Done! train time: {time.time() - train_time}")


if __name__ == '__main__':
    main()    