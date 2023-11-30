import argparse

class Config():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):

        self.parser.add_argument('--task', type=str, default='Arousal', help='the type of classification task')
        self.parser.add_argument('--path_eeg', type=str, default='data\eeg.npy', help='path of eeg modality data')
        self.parser.add_argument('--path_face', type=str, default='data\\face.npy', help='path of face modality data')
        self.parser.add_argument('--path_label', type=str, default='data\label2.npy', help='path of label. Valence——label1; Arousal——label2')
        self.parser.add_argument('--batch-size', type=int, default=40, metavar='N', help='input batch size for training [default: 40]')
        self.parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train [default: 200]')
        self.parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
        self.parser.add_argument('--alpha', type=float, default=0.5, help='alpha value of loss funciton')
        self.parser.add_argument('--beta', type=float, default=0.4, help='beta value of loss funciton')
        self.parser.add_argument('--spl_lambda', type=float, default=1, help='lambda value of spl process')
        self.parser.add_argument('--spl_gamma', type=float, default=1.15, help='gamma value of spl process')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt

