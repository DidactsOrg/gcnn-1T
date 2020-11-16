import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear, Sequential, ReLU, Dropout

class XENON_GCNN(torch.nn.Module):
    FINAL_OUT = 28 # Number of features by the end of the graph convolution phase
    # TODO: Consider making $FINAL_OUT$ a part of the input of __init__
    def __init__(self):
        super(XENON_GCNN, self).__init__()
        self.lin = Sequential(Linear(FINAL_OUT*127, 16),
                              ReLU(),
                              Linear(16, 2))
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, FINAL_OUT)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = x.view(-1, FINAL_OUT*127) # Make it into a vector for the linear operations
        x = self.lin(x)
        return x
