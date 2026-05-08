
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

# --- LOSS FUNCTIONS ---

def edl_loss(output, target, epoch, total_epochs, num_classes):
    evidence = F.relu(output)
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    probs = alpha / S
    
    target_one_hot = torch.eye(num_classes).to(target.device)[target]
    
    # Bayes Risk
    sq_diff = (target_one_hot - probs) ** 2
    variance = probs * (1 - probs) / (S + 1)
    loss_risk = torch.sum(sq_diff + variance, dim=1)
    
    # KL Regularization (Annealed)
    annealing_coef = min(1, epoch / 10)
    alpha_tilde = target_one_hot + (1 - target_one_hot) * alpha
    
    # Manual KL calculation for Dirichlet
    beta = torch.ones_like(alpha)
    S_alpha = torch.sum(alpha_tilde, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    
    lnB = torch.lgamma(alpha_tilde).sum(dim=1, keepdim=True) - torch.lgamma(S_alpha)
    lnB_uni = torch.lgamma(beta).sum(dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(alpha_tilde)
    dg1 = torch.digamma(S_alpha)
    
    kl = lnB_uni - lnB + ((alpha_tilde - beta) * (dg0 - dg1)).sum(dim=1, keepdim=True)
    
    return torch.mean(loss_risk + annealing_coef * kl.squeeze())

def orthogonality_loss(z_l, z_s):
    z_l_norm = F.normalize(z_l, p=2, dim=1)
    z_s_norm = F.normalize(z_s, p=2, dim=1)
    corr = torch.mm(z_l_norm.t(), z_s_norm)
    return torch.norm(corr, p='fro')

# --- METRIC LOGGER ---

class ResearchMetricsLogger:
    def __init__(self, n_classes=7):
        self.n_classes = n_classes
        self.reset()

    def reset(self):
        self.y_true = []
        self.y_pred_probs = []
        self.s_true = []

    def update(self, logits, targets, sensitive_attrs):
        evidence = F.relu(logits)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        probs = alpha / S
        
        self.y_true.extend(targets.cpu().numpy())
        self.y_pred_probs.extend(probs.detach().cpu().numpy())
        self.s_true.extend(sensitive_attrs.cpu().numpy())

    def get_report(self):
        y_true = np.array(self.y_true)
        y_probs = np.array(self.y_pred_probs)
        y_pred = np.argmax(y_probs, axis=1)
        s_true = np.array(self.s_true)
        
        report = {}
        # Global
        report['Global_ACC'] = accuracy_score(y_true, y_pred)
        try:
            report['Global_AUC'] = roc_auc_score(y_true, y_probs, multi_class='ovr')
        except:
            report['Global_AUC'] = 0.0
            
        # Fairness (Light=0, Dark=1)
        for g, name in [(0, 'Light'), (1, 'Dark')]:
            mask = (s_true == g)
            if np.sum(mask) > 0:
                report[f'{name}_ACC'] = accuracy_score(y_true[mask], y_pred[mask])
                # Sensitivity proxy
                cm = confusion_matrix(y_true[mask], y_pred[mask], labels=range(self.n_classes))
                with np.errstate(divide='ignore', invalid='ignore'):
                    sens = np.diag(cm) / np.sum(cm, axis=1)
                    report[f'{name}_Sens'] = np.nanmean(sens)
        
        # EOD
        sens_L = report.get('Light_Sens', 0)
        sens_D = report.get('Dark_Sens', 0)
        report['EOD'] = abs(sens_L - sens_D)
        
        return report
