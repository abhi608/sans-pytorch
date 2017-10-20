import torch.nn as nn
from torch.autograd import Variable


class QuestionEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, rnn_size, num_layers, dropout, seq_length):
        super(QuestionEmbedding, self).__init__() # Must call super __init__()

        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.lookuptable = nn.Linear(vocab_size, emb_size, bias=False)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

        self.LSTM = nn.LSTM(input_siz=emb_size, hidden_size=hidden_size,
                num_layers=num_layers, bias=True,
                batch_first=True, dropout=dropout)

        return

    def forward(self, ques_vec):            # ques_vec: [batch_size, 26]
        B, W = ques_vec.size()
        one_hot_vec = torch.zeros(B, self.vocab_size, W)

        x = []
        for i in xrange(B):
            for j in xrange(W):

                # TODO Assuming word IDs to be 1 indexed
                one_hot_vec[i][ques_vec[i][j] - 1][j] = 1

            # one_hot_vec: [batch_size, vocab_size, 26]

            x += [self.lookuptable(torch.t(emb_vec[i]))]

        # emb_vec: [batch_size or B, 26 or W, emb_size]
        emb_vec = self.dropout.(self.tanh(torch.stack(x)))


        # h: [batch_size or B, 26 or W, hidden_size]
        h, _ = self.LSTM(emb_vec)

        # TODO Understand and Implement Masking
        # h: [B, hidden_size, W]
        return torch.transpose(h, 1, 2)

