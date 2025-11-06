import torch
from torch import nn


class ReadoutNet(nn.Module):
    """
    A FC feedforward network with to be used as a readout network.

    Args:
        nodes (list): list of integers, the number of nodes in each layer.
        loss_fn (str): the loss function to be used.
        use_bias (bool): whether to use bias in the layers.

    Returns:
        torch.nn.Module: the readout network.
    """

    def __init__(self, nodes, loss_fn, use_bias=True):
        super().__init__()
        self.nodes = nodes
        if loss_fn == "crossentropy":
            self.loss_fn = nn.CrossEntropyLoss()
        self.total_params = 0
        layers = []
        for i in range(len(nodes) - 1):
            layers.append(nn.Linear(nodes[i], nodes[i + 1], bias=use_bias))
            self.total_params += (nodes[i] * nodes[i + 1]) + nodes[i + 1]
        self.fc = nn.Sequential(*layers)

    def forward(self, val):
        """
        Forward pass through the network.
        """
        return self.fc(val)

    def train_batch(self, pc_batch, label_batch, optimizer):
        """
        Train the network on a batch of pc_data.
        """
        with torch.set_grad_enabled(True):
            preds = self.forward(pc_batch)
            loss = self.loss_fn(preds, label_batch.argmax(dim=1))
            loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return preds, loss.item()

    def test_batch(self, pc_batch):
        """
        Test the network on a batch of pc_data.
        """
        with torch.no_grad():
            preds = self.forward(pc_batch)
        return preds

    def __str__(self):
        return f"<ReadoutNetwork> {self.nodes}"
