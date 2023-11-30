import torch
import statistics
import torch.nn as nn
from params import Config
from utils import SetSeeds
from net import net
from dataset import DeapDataset
from train_validate import train
from loss_fuction import MySpl_loss
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

if __name__ == '__main__':
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  

    seed = 3407
    print('seed is {}'.format(seed))
    print('training on:', device)
    SetSeeds(seed)

    args = Config().parse()

    dataset = DeapDataset(path_view1 = args.path_eeg, path_view2 = args.path_face, path_label = args.path_label)

    K = 18
    KF = KFold(n_splits=K, shuffle=False)

    predict_acc, predict_f1 = [], []

    fold = 1
    for train_idx, test_idx in KF.split(dataset):
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_subsampler)
    
        model = net(eeg_dim=160, face_dim=29, 
                    hidden_size=256, num_layers=1, 
                    dim=256, heads=1, dim_head=256, 
                    mlp_dim=512, num_classes=2, dropout=0.5).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        
        criterion1 = nn.CrossEntropyLoss()
        criterion2 = MySpl_loss(n_samples=len(train_loader.dataset), batch_size=args.batch_size, alpha=args.alpha, beta=args.beta, spl_lambda=args.spl_lambda, spl_gamma=args.spl_gamma)
        criterions = [criterion1, criterion2]

        acc, f1, train_acces, train_losses, valid_acces, valid_losses = train(model, criterions, optimizer, scheduler, train_loader, val_loader, device, args.epochs, args.alpha, args.beta, fold, args.task)

        predict_acc.append(acc)
        predict_f1.append(f1)

        print('Best of Fold {}: acc: {}, f1: {}'.format(fold, acc, f1))

        fold += 1
    
    print('Accuracy: {}'.format(sum(predict_acc) / K))
    print('F1: {}'.format(sum(predict_f1) / K))
    
    print(predict_acc)
    print(predict_f1)
    print(statistics.stdev(predict_acc))
    print(statistics.stdev(predict_f1))