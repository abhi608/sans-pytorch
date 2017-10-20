import torch.nn as nn
from torch.autograd import Variable


class ImageEmbedding(nn.Module):
    def __init__(self, hidden_size, img_features=512):
        super(ImageEmbedding, self).__init__() # Must call super __init__()

        self.hidden_size = hidden_size
        self.img_features = img_features
        self.linear = nn.Linear(img_features, hidden_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)

    def forward(self, input):
        # input: [batch_size, 196, 512]

        intermed = self.linear(input.view(-1,self.img_features)).view(
                                    -1, 196, self.hidden_size)
        return self.dropout(self.tanh(intermed))

