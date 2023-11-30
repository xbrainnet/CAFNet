""" Componets of the model
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.net(x)
        x = self.norm(x)
        return x
    
class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(MyLSTM, self).__init__()
        self.lstm = nn.Sequential(nn.LSTM(input_size, hidden_size, num_layers, batch_first=True))

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        return out

class MCLSTM(nn.Module):
    def __init__(self, input_size1, input_size2, hidden_size, num_layers, dropout):
        super(MCLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(input_size1, hidden_size)
        self.linear2 = nn.Linear(input_size2, hidden_size)
        self.shared_lstm1 = MyLSTM(hidden_size, hidden_size, num_layers)
        self.shared_lstm2 = MyLSTM(hidden_size, hidden_size, num_layers)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, eeg, face):
        eeg = self.linear1(eeg)
        face = self.linear2(face)

        h = torch.zeros((1, eeg.size(0), self.hidden_size), dtype=eeg.dtype, device=eeg.device)
        c_e = torch.zeros((1, eeg.size(0), self.hidden_size), dtype=eeg.dtype, device=eeg.device)
        c_f = torch.zeros((1, face.size(0), self.hidden_size), dtype=face.dtype, device=face.device)

        eeg, _ = self.shared_lstm1(eeg, (h, c_e))
        face, _ = self.shared_lstm2(face, (h, c_f))

        eeg = self.dropout(eeg)
        face = self.dropout(face)
        return eeg, face
    
class Attention(nn.Module):
    def __init__(self, dim, dim_head, heads):
        super(Attention, self).__init__()
        self.to_Q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_K = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_V = nn.Linear(dim, dim_head * heads, bias=False)
        self.norm = nn.LayerNorm(dim)

    def attention(self, Q, K, V):
        d_k = K.size(-1)
        scores = torch.matmul(Q, K.transpose(1,2)) / math.sqrt(d_k)
        alpha_n = F.softmax(scores, dim=-1)
        output = torch.matmul(alpha_n, V)
        output = output.sum(1)
        return output, alpha_n

    def forward(self, x):
        Q = self.to_Q(x)
        K = self.to_K(x)
        V = self.to_V(x)
        out, _ = self.attention(Q, K, V)
        out = self.norm(out)
        return out

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))

    def forward(self, x):
        x = self.clf(x)
        return x

class EegSubNet(nn.Module):
    def __init__(self, dropout, dim, heads, dim_head, mlp_dim):
        super(EegSubNet, self).__init__()
        self.SelfAttention = Attention(dim, heads, dim_head)
        self.FeedForward = FeedForward(dim, mlp_dim, dropout)
    
    def forward(self, x):
        x = self.SelfAttention(x)
        x = self.FeedForward(x)
        return x

class FaceSubNet(nn.Module):
    def __init__(self, dropout, dim, heads, dim_head, mlp_dim):
        super(FaceSubNet, self).__init__()
        self.SelfAttention = Attention(dim, heads, dim_head)
        self.FeedForward = FeedForward(dim, mlp_dim, dropout)
    
    def forward(self, x):
        x = self.SelfAttention(x)
        x = self.FeedForward(x)
        return x
    
class RegressionSubNetwork(nn.Module):
    def __init__(self, mlp_dim):
        super(RegressionSubNetwork, self).__init__()
        self.layers = nn.ModuleList([LinearLayer(mlp_dim, 1)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ClassificationSubNetwork(nn.Module):
    def __init__(self, mlp_dim, num_classes):
        super(ClassificationSubNetwork, self).__init__()
        self.layers = nn.ModuleList([LinearLayer(mlp_dim, num_classes)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class net(nn.Module):  
    def __init__(self, eeg_dim, face_dim, hidden_size, num_layers, dim, heads, dim_head, mlp_dim, num_classes, dropout):
        super().__init__()
        self.num_classes = num_classes
        self.mc_lstm = MCLSTM(eeg_dim, face_dim, hidden_size, num_layers, dropout)
        self.eeg_subnet = EegSubNet(eeg_dim, hidden_size, num_layers, dropout, dim, heads, dim_head, mlp_dim)
        self.face_subnet = FaceSubNet(face_dim, hidden_size, num_layers, dropout, dim, heads, dim_head, mlp_dim)
        self.Regression = RegressionSubNetwork(dim)
        self.Classification = ClassificationSubNetwork(dim, num_classes)
        self.fc = nn.Linear(mlp_dim, num_classes)
        
    def confidence_loss(self, TCPLogit, TCPConfidence, label):
        pred = F.softmax(TCPLogit, dim=1)
        p_target = torch.gather(input=pred, dim=1, index=label.unsqueeze(dim=1).type(torch.int64)).view(-1)
        c_loss = torch.mean(F.mse_loss(TCPConfidence.view(-1), p_target, reduction='none'))
        return c_loss

    def KD_loss(self, TCPLogit_eeg, TCPLogit_face):
        loss1 = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(TCPLogit_eeg, dim=1), F.softmax(TCPLogit_face, dim=1))
        loss2 = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(TCPLogit_face, dim=1), F.softmax(TCPLogit_eeg, dim=1))
        return (loss1 + loss2) / 2
    
    def forward(self, v, label):
        eeg, face = v
        eeg, face = self.mc_lstm(eeg, face)
        eeg = self.eeg_subnet(eeg)
        face = self.face_subnet(face)
        
        TCPConfidence_eeg = self.Regression(eeg)
        TCPConfidence_face = self.Regression(face)
        TCPLogit_eeg = self.Classification(eeg)
        TCPLogit_face = self.Classification(face)

        eeg = eeg * TCPConfidence_eeg
        face = face * TCPConfidence_face
        
        feature = torch.cat([eeg, face], dim=1)
        Logit = self.fc(feature)
        
        c_loss_eeg = self.confidence_loss(TCPLogit_eeg, TCPConfidence_eeg, label)
        c_loss_face = self.confidence_loss(TCPLogit_face, TCPConfidence_face, label)
        c_loss = c_loss_eeg + c_loss_face
        kd_loss = self.KD_loss(TCPLogit_eeg, TCPLogit_face)
        return Logit, c_loss, kd_loss
