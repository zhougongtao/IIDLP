import torch
from torch import nn
from torch.autograd import Variable
from torch.distributions import Normal
import torch.nn.functional as F

"""
    SimpleNN

"""
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )
        self.lam_layer = nn.Linear(16, 1)
        self.risk_layer = nn.Linear(16, 1)
    
    def forward(self, x):
        shared = self.shared_layer(x) 
        lam = self.lam_layer(shared)
        risk = self.risk_layer(shared)
        return lam, risk

""" 
    BNN
        BayesianLinearWithMOPED
        BayesianNetworkWithMOPED
    
"""
    
def get_rho(sigma, delta):
    """
        sigma = log(1 + exp(rho)) 
    """
    rho = torch.log(torch.expm1(delta * torch.abs(sigma)) + 1e-20)
    return rho


def kl_div(mu_q, sigma_q, mu_p, sigma_p):
    """
        Calculates kl divergence between two gaussians (Q || P)
    """
    kl = torch.log(sigma_p) - torch.log(sigma_q) + (sigma_q**2 + (mu_q - mu_p)**2) / (2 *(sigma_p**2)) - 0.5
    return kl.mean()


class BayesianLinearWithMOPED(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # prior
        self.register_buffer('prior_w_mu', torch.Tensor(out_features, in_features), persistent=False)
        self.register_buffer('prior_w_sigma', torch.Tensor(out_features, in_features), persistent=False)
        self.register_buffer('prior_b_mu', torch.Tensor(out_features), persistent=False)
        self.register_buffer('prior_b_sigma', torch.Tensor(out_features), persistent=False)
        
        # epsilon
        self.register_buffer('w_eps', torch.Tensor(out_features, in_features), persistent=False)
        self.register_buffer('b_eps', torch.Tensor(out_features), persistent=False)
        
        # posterior
        self.w_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.w_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.b_mu = nn.Parameter(torch.Tensor(out_features))
        self.b_rho = nn.Parameter(torch.Tensor(out_features))       
    
    def init_parameters(self, layer, delta, prior_var):
        # prior
        self.prior_w_mu.data = layer.weight.data.clone().detach().requires_grad_(True)
        self.prior_w_sigma.fill_(prior_var)
        self.prior_b_mu.data = layer.bias.data.clone().detach().requires_grad_(True)
        self.prior_b_sigma.fill_(prior_var)
        
        # posterior
        self.w_mu.data = layer.weight.data.clone().detach().requires_grad_(True)
        self.w_rho.data = get_rho(self.w_mu.data, delta)
        self.b_mu.data = layer.bias.data.clone().detach().requires_grad_(True)
        self.b_rho.data = get_rho(self.b_mu.data , delta)

    def forward(self, input, infer=False):
        if infer:
            return F.linear(input, self.w_mu, self.b_mu)
        
        w_sigma = torch.log1p(torch.exp(self.w_rho))
        self.w = self.w_mu + w_sigma * self.w_eps.data.normal_()

        b_sigma = torch.log1p(torch.exp(self.b_rho))
        self.b = self.b_mu + b_sigma * self.b_eps.data.normal_()

        w_kl = kl_div(self.w_mu, w_sigma, self.prior_w_mu, self.prior_w_sigma)
        b_kl = kl_div(self.b_mu, b_sigma, self.prior_b_mu, self.prior_b_sigma)
        self.kl = w_kl + b_kl
        
        return F.linear(input, self.w, self.b)
    
    def kl_EM(self):
        w_sigma = torch.log1p(torch.exp(self.w_rho))
        b_sigma = torch.log1p(torch.exp(self.b_rho))
        w_kl = kl_div(self.w_mu, w_sigma, self.prior_w_mu, self.prior_w_sigma)
        b_kl = kl_div(self.b_mu, b_sigma, self.prior_b_mu, self.prior_b_sigma)
        return w_kl + b_kl
        
class BayesianNetworkWithMOPED(nn.Module):
    def __init__(self, model, delta=0.5, prior_var=1.0, design=None):
        super().__init__()
        
        self.hidden1 = BayesianLinearWithMOPED(16, 16)
        self.hidden2 = BayesianLinearWithMOPED(16, 16)
        self.lam_layer = BayesianLinearWithMOPED(16, 1)
        self.risk_layer = BayesianLinearWithMOPED(16, 1)
        
        self.hidden1.init_parameters(model.shared_layer[0], delta, prior_var)
        self.hidden2.init_parameters(model.shared_layer[2], delta, prior_var)
        self.lam_layer.init_parameters(model.lam_layer, delta, prior_var)
        self.risk_layer.init_parameters(model.risk_layer, delta, prior_var)

    def forward(self, x, infer=False):
        x = F.relu(self.hidden1(x, infer))
        x = F.relu(self.hidden2(x, infer))
        lam = self.lam_layer(x)
        risk = self.risk_layer(x)
        return lam, risk

    def kl(self):
        return self.hidden1.kl + self.hidden2.kl + self.lam_layer.kl + self.risk_layer.kl
    
    def kl_EM(self):
        return self.hidden1.kl_EM() + self.hidden2.kl_EM() + self.out.kl_EM()


"""
    EarlyStoppping

"""
class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_design_idx = None
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.counter = 1
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 1