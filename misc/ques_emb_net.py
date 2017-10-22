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

    def forward(self, ques_vec, ques_len):            # ques_vec: [batch_size, 26]
        B, W = ques_vec.size()
        one_hot_vec = torch.zeros(B, self.vocab_size, W)

        x = []
        for i in xrange(B):
            for j in xrange(W):

                if not ques_vec[i][j]:
                    break

                one_hot_vec[i][ques_vec[i][j] - 1][j] = 1

            x += [self.lookuptable(torch.t(one_hot_vec[i]))]

        # emb_vec: [batch_size or B, 26 or W, emb_size]
        emb_vec = self.dropout.(self.tanh(torch.stack(x)))


        # h: [batch_size or B, 26 or W, hidden_size]
        h, _ = self.LSTM(emb_vec)

        # TODO Understand and Implement Masking
        # output: [B, hidden_size]
        return h[:,-1,:]

