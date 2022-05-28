import torch
import torch.nn as nn
from torch.nn import functional as F


class ActDropNormCNN1D(nn.Module):
    def __init__(self, n_feats, dropout):
        super(ActDropNormCNN1D, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.dropout(F.gelu(self.norm(x)))
        return x


class SpeechRecognition(nn.Module):

    def __init__(self, hidden_size, num_classes, n_feats, num_layers, dropout):
        super(SpeechRecognition, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.cnn = nn.Sequential(
            nn.Conv1d(n_feats, n_feats, kernel_size=(10,), stride=(2,), padding=10 // 2, dilation=(1,)),
            ActDropNormCNN1D(n_feats, dropout),
        )

        self.dense = nn.Sequential(
            nn.Linear(n_feats, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.0,
            bidirectional=False
        )

        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout)

        self.final_distribution = nn.Linear(hidden_size, num_classes)

    def forward(self, x, hidden):
        x = x.squeeze(1)

        x = self.cnn(x)
        x = self.dense(x)

        x = x.transpose(0, 1)

        out, (hidden_state, cell_state) = self.lstm(x, hidden)
        x = self.layer_norm2(out)
        x = F.gelu(x)
        x = self.dropout2(x)

        return self.final_distribution(x), (hidden_state, cell_state)

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers * 1, batch_size, self.hidden_size),
                torch.zeros(self.num_layers * 1, batch_size, self.hidden_size))
