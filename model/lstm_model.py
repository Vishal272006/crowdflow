"""
lstm_model.py

Dual-head LSTM for CrowdFlow bunching prediction (Phase 2).

Architecture (matches the PPT "Architecture - LSTM + RL Hybrid" slide):
    Input sequence -> LSTM (temporal sequencing) -> shared representation
        -> Regression head:     headway deviation (minutes) at next H stops
        -> Classification head: bunching risk class (Low/Med/High) at next H stops

The two heads share the LSTM's learned temporal representation because
they're two views of the same underlying signal -- the classification
head is a simple, easy-to-monitor readout used to trigger the RL layer
(Phase 3), while the regression head gives a precise magnitude estimate.
"""

import torch
import torch.nn as nn


class CrowdFlowLSTM(nn.Module):
    def __init__(self, n_features, hidden_size=64, horizon=3, n_classes=3,
                 num_layers=1, dropout=0.0):
        super().__init__()
        self.horizon = horizon
        self.n_classes = n_classes

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Regression head: predicts continuous headway deviation (minutes)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, horizon),
        )

        # Classification head: predicts risk class logits for each of the
        # next `horizon` stops -- output reshaped to (batch, horizon, n_classes)
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, horizon * n_classes),
        )

    def forward(self, x):
        # x: (batch, window, n_features)
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # (batch, hidden_size) -- final layer's hidden state

        reg_out = self.regression_head(last_hidden)               # (batch, horizon)
        cls_out = self.classification_head(last_hidden)
        cls_out = cls_out.view(-1, self.horizon, self.n_classes)  # (batch, horizon, n_classes)

        return reg_out, cls_out
