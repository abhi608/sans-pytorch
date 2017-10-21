class CDATA(torch.utils.data.Dataset): # Extend PyTorch's Dataset class
    def __init__(self, opt, train, transform=None):
        print('DataLoader loading h5 question file: ' + opt['h5_ques_file'])
        h5_file = h5py.File(opt['h5_ques_file'], 'r')
        if train:
            print('DataLoader loading h5 image train file: ' + opt['h5_img_file'])
            self.h5_img_file = h5py.File(opt['h5_img_file'], 'r')
            self.ques = h5_file['/ques_train']
            self.ques_len = h5_file['/ques_len_train']
            self.img_pos = h5_file['/img_pos_train']
            self.ques_id = h5_file['/ques_id_train']
            self.ans = h5_file['/answers']
            self.split = h5_file['/split_train']
        else:
            print('DataLoader loading h5 image test file: ' + opt['h5_img_file'])
            self.h5_img_file = h5py.File(opt['h5_img_file'], 'r')
            self.ques = h5_file['/ques_test']
            self.ques_len = h5_file['/ques_len_test']
            self.img_pos = h5_file['/img_pos_test']
            self.ques_id = h5_file['/ques_id_test']
            self.ans = h5_file['/ans_test']
            self.split = h5_file['/split_test']

        h5_file.close()
        self.feature_type = opt['feature_type']
        self.train = train
        self.transform = transform
        
    def __len__(self):

        return self.split.shape[0]
        
    def __getitem__(self, idx):

        img_idx = self.img_pos[idx]
        if self.h5_img_file:
            if train:
                if self.feature_type == 'VGG':
                    img = self.h5_img_file['/images_train'][img_idx, 0:14, 0:14, 0:512]  # [14, 14, 512]
                elif self.feature_type == 'Residual':
                    img = self.h5_img_file['/images_train'][img_idx, 0:14, 0:14, 0:2048] # [14, 14, 2048]
                else:
                    print("Error(train): feature type error")
            else:
                if self.feature_type == 'VGG':
                    img = self.h5_img_file['/images_test'][img_idx, 0:14, 0:14, 0:512] # [14, 14, 512]
                elif self.feature_type == 'Residual':
                    img = self.h5_img_file['/images_test'][img_idx, 0:14, 0:14, 0:2048] # [14, 14, 2048]
                else:
                    print("Error(test): feature type error")

        questions = self.ques[idx] # vector of size 21
        ques_id = self.ques_id[idx] # scalar integer
        ques_len = self.ques_len[idx] # scalar integer
        answer = self.ans[idx] # scalar integer
        return (img, questions, ques_id, ques_len, answer)