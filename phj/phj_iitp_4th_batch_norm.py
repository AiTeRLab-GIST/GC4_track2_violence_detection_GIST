#!/usr/bin/env python
# coding: utf-8
import random
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler
import os
import argparse
from sklearn.metrics import confusion_matrix

os.environ["TOKENIZERS_PARALLELISM"] = "false"
parser = argparse.ArgumentParser(description='path')
parser.add_argument('--path_dir_in', type = str,required=True, default=None, help='input_dir')
parser.add_argument('--path_dir_out', type= str, required=True, default=None, help='out_dir')
#parser.add_argument('--file_name', type= str, required=True, default=None, help='file_name')
args = parser.parse_args()
path_in = args.path_dir_in
path_out = args.path_dir_out
#file_name = args.file_name
file_name_index = "phj_result.txt"
file_name_weight = "phj_weight.txt"
print("path_in:", path_in)
print("path_out:", path_out)
print("file_name_index", file_name_index)
print("file_name_weight", file_name_weight)
#path_in = os.path.abspath(path_in)
#path_out = os.path.abspath(path_out)
print("path_in", path_in)
random.seed(4)
torch.manual_seed(4)
torch.cuda.manual_seed(4)
torch.cuda.manual_seed_all(4)

#device = torch.device("cpu")
device = torch.device("cuda")
bertmodel, vocab = get_pytorch_kobert_model()
tokenizer = get_tokenizer()  ##gives path
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False) 




## Setting parameters
max_len = 329
batch_size = 1


import torch
import torch.nn as nn
class LSTMClassfier_batchnorm(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=5,
                 dr_rate=None,
                 params=None,
                 bidirectional=None,
                num_layers = 2):
        
        super(LSTMClassfier_batchnorm, self).__init__()
        self.num_classes = num_classes
        self.bert = bert  ## batch, length, hidden
        embedding_dim = bert.config.to_dict()['hidden_size'] # 768
        self.LSTM = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=2,
                            bias = True, dropout=0.5, bidirectional = True, batch_first = True)
        self.dr_rate = dr_rate
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dr_rate)
            
        self.linear1 = nn.Linear(hidden_size*4, 4)
        self.linear = nn.Linear(hidden_size*2, num_classes)
        self.classifier = nn.Softmax(dim=1)
        
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        #print(len(valid_length))
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        #print("attention",attention_mask)
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        #print("token_ids",token_ids)
        #print("segment_ids", segment_ids)
        last_hidden_state, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        out, hidden = self.LSTM(last_hidden_state)
        out = self.dropout(out)
        print("out", out.shape)
        out = self.linear(out)
        #out = self.linear1(out)

        #out = self.linear1(out)
        return self.classifier(out[:,-1,:]/1.5)

model = LSTMClassfier_batchnorm(bertmodel,  dr_rate=0.5).to(device)


# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

#warmup_step = int(t_total * warmup_ratio)

#scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)
    
path_subdir = path_out.split('/')
if(not(os.path.isdir(os.path.join(path_out,path_subdir[-1])))):
    directory_path = os.path.join(path_out)
    try:
        os.mkdir(directory_path.replace('\r',''))
    except:
        print('directory already exits, saving files in' +os.path.join(directory_path, file_name_weight))


#print(os.listdir(path_in))

list_of_files = sorted(os.listdir(path_in))

print(len(list_of_files))


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, file_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        #self.files = [np.array(i[file_idx]) for i in dataset]
    def __getitem__(self, i):
        return (self.sentences[i])
    def __len__(self):
        return (len(self.sentences))
        



torch.set_printoptions(precision=10)
from sklearn.metrics import f1_score
import os.path
from os import path

model = LSTMClassfier_batchnorm(bertmodel,  dr_rate=0.5).to(device)
model.load_state_dict(torch.load('./phj/original_203_overfit_batch_norm_LSTM_768.pt'))

f_out = open(path_out+'bert_text.txt','w')
for file in sorted(list_of_files):
    print("file",file)
    in_file = open(path_in+"/"+file, "r")
    print("file_name", in_file.name)
    utterance = ""
    lines = in_file.readlines()
    for line in lines:
        onset = line.split('\t')[1]
        offset = line.split('\t')[2]
        text = line.split('\t')[0]
        utterance+= text.replace('\n','')+' '
    print("utterance", utterance)
    f_out.writelines(utterance+'\t'+file+'\n')
f_out.close()
max_len = 329
#input_text = (os.path.abspath('/home/data2/phj/iitp_data/cross/'+'iitp_0_train_5_ref'))
input_text = (os.path.abspath(path_out+'bert_text.txt'))
#dataset_test = nlp.data.TSVDataset("./waves_sample/iitp_sample_with_labels.txt", field_indices = [0] )
dataset_test = nlp.data.TSVDataset(input_text, field_indices = [0])
bert_dataset = BERTDataset(dataset_test,0, 1, tok, max_len, True, False)
counter=0

test_dataloader = torch.utils.data.DataLoader(bert_dataset, batch_size=batch_size, num_workers=5)  

confusion_label = []
confusion_out = []
out_index = open(os.path.join(path_out,file_name_index),"w+")
out_weight = open(os.path.join(path_out,file_name_weight),"w+")
model.eval()
with torch.no_grad():
    for batch_id, (token_ids, valid_length, segment_ids) in enumerate(tqdm(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        out = model(token_ids, valid_length, segment_ids)
        files = list_of_files[counter]
        print("counter", counter, out)
        values,indices = torch.max(out,1)        
        index = indices.cpu().detach().numpy()[0]
        softmax = values.cpu().detach().numpy()
        softmax_arr = out.cpu().detach().numpy()[0]
        print("softmax_arr",softmax_arr)
        out_index.writelines(files.replace('txt','wav')+'\t'+str(index)+'\n')
        out_weight.writelines(files.replace('txt','wav')+'\t'+str(softmax_arr[0])+'\t'+str(softmax_arr[1])+'\t'+str(softmax_arr[2])+'\t'+str(softmax_arr[3])+'\n')
        counter+=1
        
out_weight.close()
out_index.close()





