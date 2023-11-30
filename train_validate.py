import time
import torch
import torch.nn as nn
from utils import Metrics

def train(model, criterions, optimizer, scheduler, train_loader, valid_loader, device, num_epoch, alpha, beta, fold, task, init=False):
    def init_kaiming(m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight.data)

    if init:
        model.apply(init_kaiming)

    model.to(device)
    
    best_acc, f1 = 0.0, 0.0
    train_acces, train_losses, valid_acces, valid_losses = [], [], [], []

    for epoch in range(num_epoch):
        start = time.time()
        model.train()
        train_epoch_loss = 0
        valid_epoch_loss = 0

        # train
        predict_train, y_true_train = None, None
        for i, (v, labels, index) in enumerate(train_loader):
            v[0] = v[0].to(device)
            v[1] = v[1].to(device)
            label = labels.view(-1).to(device)

            predict_train, c_loss, kd_loss = model(v, label)

            loss = criterions[1](predict_train, label.long(), i, c_loss, kd_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_epoch_loss += loss.item()

            predict_train = predict_train.max(1)[1]

            if i == 0:
                pre = predict_train
                y_true_train = label
            else:
                pre = torch.hstack((pre, predict_train))
                y_true_train = torch.hstack((y_true_train, label))
            
        scheduler.step()
        criterions[1].increase_threshold()
        train_acc, _ = Metrics(y_true_train.cpu(), pre.cpu())
        
        # valid
        predict, y_true = None, None
        model.eval()       
        with torch.no_grad():
            for i, (v, labels, index) in enumerate(valid_loader):
                v[0] = v[0].to(device)
                v[1] = v[1].to(device)
                label = labels.view(-1).to(device)

                predict, c_loss, kd_loss = model(v, label)

                loss = criterions[0](predict, label.long()) + alpha * c_loss + beta * kd_loss
                
                valid_epoch_loss += loss.item()

                predict = predict.max(1)[1]

                if i == 0:
                    pre = predict
                    y_true = label
                else:
                    pre = torch.hstack((pre, predict))
                    y_true = torch.hstack((y_true, label))

            valid_acc, valid_f1 = Metrics(y_true.cpu(), pre.cpu())
        
            if valid_acc > best_acc:
                best_acc = valid_acc
                f1 = valid_f1
                torch.save(model.state_dict(), 'checkpoint/' + task + '/Fold_' + str(fold) + '_best_acc.pth')
            
            train_epoch_loss = train_epoch_loss / len(train_loader)
            valid_epoch_loss = valid_epoch_loss / len(valid_loader)

            end = time.time() - start

        train_acces.append(train_acc)
        train_losses.append(train_epoch_loss)
        valid_acces.append(valid_acc)
        valid_losses.append(valid_epoch_loss)
        print("< Fold{} {:.0f}% {}/{} {:.3f}s >".format(fold, (epoch + 1) / num_epoch * 100, epoch + 1, num_epoch, end), end="")
        print('train_loss =', '{:.5f}'.format(train_epoch_loss), end=" ")
        print('train_acc =', '{:.5f}'.format(train_acc), end=" ")
        print('valid_loss =', '{:.5f}'.format(valid_epoch_loss), end=" ")
        print('valid_acc =', '{:.4f}'.format(valid_acc))
    
    return best_acc, f1, train_acces, train_losses, valid_acces, valid_losses
