import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

def SetSeeds(seed):
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.set_printoptions(precision=8)
    np.random.seed(seed)

def Metrics(y_true, pre):
    acc, f1 = 0.0, 0.0
    try:
        ACC = accuracy_score(y_true, pre)
        F1 = f1_score(y_true, pre)

        acc += ACC
        f1 += F1

    except ValueError as ve:
        print(ve)
        pass

    return acc, f1

    