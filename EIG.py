import torch
from data.InvestorContext import *

# y is finite, enumerating the type of y 
def nmc_eig_y(model, x, design, args):
    model.eval()
    with torch.no_grad():
        batch = x.shape[0]
        EIG = torch.zeros(design.shape[0])

        for d_idx, d in enumerate(design):
            marginal = torch.zeros(batch, args.portfolio_num)
            pro_save = torch.zeros(batch, args.portfolio_num, args.NMC_L)

            for count_sample in range(args.NMC_L):
                lam_j, risk_j = model(x)
                log_prob_j = prob_portfolio(lam_j, risk_j, d)
                prob_j = torch.exp(log_prob_j)
                marginal += prob_j
                pro_save[:, :, count_sample] = prob_j
            marginal /= args.NMC_L

            marginal_expanded = marginal.unsqueeze(2)
            ratio = pro_save / torch.clamp(marginal_expanded, min=1e-30)
            g = pro_save * torch.log(torch.clamp(ratio, min=1e-30))
            EIG[d_idx] = g.sum() / args.NMC_L

            if torch.isnan(EIG).any():
                print("Error: EIG have nan!!!")

        _, sorted_indices = torch.sort(EIG)
        middle_index = len(EIG) // 2
        middle_value_index = sorted_indices[middle_index]

        if args.strategy == 12: return torch.argmin(EIG), EIG.tolist()  # ID-LP-minEIG
        elif args.strategy == 13: return middle_value_index, EIG.tolist()  # ID-LP-medEIG
        else: return torch.argmax(EIG), EIG.tolist()  # ID-LP


# for I-ID-LP calculate logEIG
def nmc_eig_EM_fast(model, x, d, args):
    batch = x.shape[0]
    EIG = torch.zeros(1)
    
    marginal = torch.zeros(batch, args.portfolio_num)
    pro_save = torch.zeros(batch, args.portfolio_num, args.NMC_K_train_fast)

    for count_sample in range(args.NMC_K_train_fast):
        lam_j, risk_j = model(x)
        log_prob_j = prob_portfolio(lam_j, risk_j, d)
        prob_j = torch.exp(log_prob_j)
        marginal += prob_j
        pro_save[:, :, count_sample] = prob_j
    marginal /= args.NMC_K_train_fast

    marginal_expanded = marginal.unsqueeze(2)
    ratio = pro_save / torch.clamp(marginal_expanded, min=1e-30)
    g = pro_save * torch.log(torch.clamp(ratio, min=1e-30))
    g_sum = g.sum(dim=(0, 1))
    EIG_vec = torch.log(torch.clamp(g_sum, min=1e-10))
    EIG = EIG_vec.sum()  # Compute the final EIG by summing over all the samples

    EIG /= args.NMC_K_train_fast

    return EIG


def nmc_eig_EM_fast_calculate(model, x, d, args):
    batch = x.shape[0]
    EIG = torch.zeros(1)

    marginal = torch.zeros(batch, args.portfolio_num)
    pro_save = torch.zeros(batch, args.portfolio_num, args.NMC_K_train_fast)

    for count_sample in range(args.NMC_K_train_fast):
        lam_j, risk_j = model(x)
        log_prob_j = prob_portfolio(lam_j, risk_j, d)
        prob_j = torch.exp(log_prob_j)
        marginal += prob_j
        pro_save[:, :, count_sample] = prob_j
    marginal /= args.NMC_K_train_fast

    marginal_expanded = marginal.unsqueeze(2)
    ratio = pro_save / torch.clamp(marginal_expanded, min=1e-30)
    g = pro_save * torch.log(torch.clamp(ratio, min=1e-30))
    g_sum = g.sum(dim=(0, 1))
    EIG_vec = torch.log(torch.clamp(g_sum, min=1e-10))
    EIG = EIG_vec.sum()  # Compute the final EIG by summing over all the samples

    EIG /= args.NMC_K_train_fast
    return EIG.item()