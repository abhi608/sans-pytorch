import torch.nn as nn
from torch.autograd import Variable


class Attention(nn.Module): # Extend PyTorch's Module class
    def __init__(self, input_size, att_size, img_seq_size, output_size, drop_ratio):
        super(CustomResnet, self).__init__() # Must call super __init__()      
        self.fc11 = nn.Linear(input_size, att_size, bias=True)
        self.fc12 = nn.Linear(input_size, att_size, bias=False)
        self.tan1 = nn.Tanh()
        self.dp1 = nn.Dropout(drop_ratio)
        self.fc13 = nn.Linear(att_size, 1, bias=True)
        self.sf1 = nn.Softmax()

        self.fc21 = nn.Linear(input_size, att_size, bias=True)
        self.fc22 = nn.Linear(input_size, att_size, bias=False)
        self.tan2 = nn.Tanh()
        self.dp2 = nn.Dropout(drop_ratio)
        self.fc23 = nn.Linear(att_size, 1, bias=True)
        self.sf2 = nn.Softmax()

        self.fc = nn.Linear(input_size, output_size, bias=True)

        # d = input_size | m = img_seq_size
    def forward(self, ques_feat, img_feat):  # ques_feat -- [batch, d] | img_feat -- [batch_size, m, d]
        x = self.conv1(x)
        # Stack 1
        ques_emb_1 = self.fc11(ques_feat)  # [batch_size, att_size]
        
        ques_emb_expand_1 = Variable(torch.rand(batch_size, img_seq_size, att_size))
        for i in range(0,100):
            for j in range(0, 196):
                ques_emb_expand_1[i,j] = ques_emb_1[i]

        img_emb_dim_1 = nn.fc12(img_feat.view(-1, input_size))
        img_emb_1 = img_emb_dim_1.view(-1, img_seq_size, att_size)
        h1 = self.tan1(ques_emb_expand_1 + img_emb_1)
        h1_drop = self.dp1(h1)
        h1_emb = self.fc13(h1_drop.view(-1, att_size))
        p1 = self.sf1(h1_emb.view(-1, img_seq_size))
        p1_att = p1.view(batch_size, 1, img_seq_size)
        # Weighted sum
        img_att1 = p1_att.matmul(img_feat)
        img_att_feat_1 = img_att1.view(-1, input_size)
        u1 = ques_feat + img_att_feat_1

        # Stack 2
        ques_emb_2 = self.fc21(u1)  # [batch_size, att_size]
        
        ques_emb_expand_2 = Variable(torch.rand(batch_size, img_seq_size, att_size))
        for i in range(0,100):
            for j in range(0, 196):
                ques_emb_expand_2[i,j] = ques_emb_2[i]

        img_emb_dim_2 = nn.fc22(img_feat.view(-1, input_size))
        img_emb_2 = img_emb_dim_2.view(-1, img_seq_size, att_size)
        h2 = self.tan2(ques_emb_expand_2 + img_emb_2)
        h2_drop = self.dp1(h2)
        h2_emb = self.fc13(h2_drop.view(-1, att_size))
        p2 = self.sf1(h2_emb.view(-1, img_seq_size))
        p2_att = p2.view(batch_size, 1, img_seq_size)
        # Weighted sum
        img_att2 = p2_att.matmul(img_feat)
        img_att_feat_2 = img_att2.view(-1, input_size)
        u2 = u1 + img_att_feat_2

        # score
        score = self.fc(u2)

        return score




