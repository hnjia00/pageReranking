import torch.nn as nn
import torch
import argparse
import random
import numpy as np
import torch.nn.functional as F
from torchsampler import ImbalancedDatasetSampler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
import time
import os
import json

class Simulator(nn.Module):

    def __init__(self,n_input,n_hidden,n_output):
        super(Simulator, self).__init__()
        self.hidden = nn.Sequential(nn.Linear(n_input,2*n_hidden),
                                    nn.ReLU(),
                                    nn.Linear(2*n_hidden,n_hidden),
                                    # nn.Dropout(p=0.5),
                                    nn.ReLU(),
                                    nn.Linear(n_hidden,n_output))

    def forward(self, input):
        out = self.hidden(input)
        out = torch.sigmoid(out)
        return out
        
class Mydata(Dataset):
    def __init__(self, data, batch_size, mode):
        self.mode = mode

        if self.mode=='pv':
            pvs = []
            for pv in data:
                pv_data = []
                flag = False
                for item in pv[:6]:
                    if int(item[2]) == 1:
                        flag = True
                    pv_data.extend(item[3:])
                pv_data.append(1) if flag else pv_data.append(0)
                pvs.append(torch.FloatTensor(pv_data))
            self.x = pvs
            self.x = self.x[:int(len(self.x)/batch_size)*batch_size]
            # print(len(self.x))
        else:
            self.x = [torch.FloatTensor(i) for i in data]

    def __getitem__(self, idx):
        if self.mode=='pv':
            assert idx < len(self.x)
            return self.x[idx][:-1], self.x[idx][-1]
        else:
            return self.x[idx][3:], self.x[idx][1]
    
    def __getlabel__(self, idx):
        if self.mode=='pv':
            assert idx < len(self.x)
            return self.x[idx][-1].item()
        else:
            assert idx < len(self.x)
            return self.x[idx][1].item()

    def __len__(self):
        return len(self.x)
    
    def countweights(self):

        self.indices = list(range(len(self.x)))
        self.num_samples = len(self.indices)

        label_to_count = {}
        for idx in self.indices:
            label = self.__getlabel__(idx)
            label_to_count[label] = label_to_count.get(label, 0) + 1

        # weight for each sample
        weights = [1.0 / label_to_count[self.__getlabel__(idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

        return self.weights

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='pv', choices=['item', 'pv'], type=str)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--split_ratio', default=0.7, type=float)
parser.add_argument('--learning_rate', default=1e-6, type=float)
parser.add_argument('--input_size', default=28*6, type=int)
parser.add_argument('--hidden_size', default=64, type=int)
parser.add_argument('--output_size', default=1, type=int)
parser.add_argument('--model_dir', default='./model/simulator', type=str)
parser.add_argument('--data_dir', default='./data', type=str)
parser.add_argument('--train_batch_size', default=128, type=int)
parser.add_argument('--eval_batch_size', default=64, type=int)
args = parser.parse_args()

def evaluate_accuracy(outputs, labels):
    correct= (outputs.ge(0.5) == labels).sum().item()
    n = labels.shape[0]
    return correct / n

def evaluate(model, dataloader, criterion, device):
    epoch_eval_loss = []
    epoch_eval_acc = []
    TP, TN, FP, FN = 0, 0, 0, 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs).squeeze(dim=1)
            loss = criterion(outputs, labels)
            # outputs = torch.sigmoid(outputs)
            eval_acc = evaluate_accuracy(outputs, labels)

            # print(outputs)
            # print((outputs.ge(0.5) == False) & (labels == 0))
            TP += ((outputs.ge(0.5) == True) & (labels == 1)).sum().item()
            TN += ((outputs.ge(0.5) == False) & (labels == 0)).sum().item()
            FP += ((outputs.ge(0.5) == True) & (labels == 0)).sum().item()
            FN += ((outputs.ge(0.5) == False) & (labels == 1)).sum().item()
            # TN += ((outputs.ge(0.5) == True) & (labels == 1)).sum().item()
            # TP += ((outputs.ge(0.5) == False) & (labels == 0)).sum().item()
            # FN += ((outputs.ge(0.5) == True) & (labels == 0)).sum().item()
            # FP += ((outputs.ge(0.5) == False) & (labels == 1)).sum().item()

            total += (labels == 1).sum().item()
            epoch_eval_loss.append(loss.item())
            epoch_eval_acc.append(eval_acc)
        
        print(TP, total)
        p = TP/max(TP+FP,1e-4)
        r = TP/max(TP+FN,1e-4)
        f1 = 2*p*r/max((p+r),1e-4)
    # print(TP, TN, FP, FN)
    return epoch_eval_loss,  epoch_eval_acc, f1

def train(model, dataloader, optimizer, criterion, device):
    epoch_train_loss = []
    epoch_train_acc = []
    for inputs, labels in dataloader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs).squeeze(dim=1)
        loss = criterion(outputs, labels)
        # outputs = torch.sigmoid(outputs)
        train_acc = evaluate_accuracy(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_loss.append(loss.item())
        epoch_train_acc.append(train_acc)      

    return epoch_train_loss, epoch_train_acc
    

if __name__=='__main__':
    device = 'cpu'
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # torch.cuda.set_device(1)
    print(device)
    

    if args.mode == 'pv':
        with open(os.path.join(args.data_dir, 'filtered_day1.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)
        random.shuffle(data)
        print(len(data))
    else:
        with open(os.path.join(args.data_dir, 'filtered_day1.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)
        random.shuffle(data)
        data = [item for pv in data for item in pv[:6]]
        print(len(data))
        
    model = Simulator(args.input_size, args.hidden_size, args.output_size)
    model = model.to(device)
    
    offset = int(len(data) * args.split_ratio)
    train_data = data[:offset]
    eval_data = data[offset:]
    
    s = 0
    for i in eval_data:
        if i[2]==1:
            s+=1
    print(len(eval_data), s)

    train_dataset = Mydata(train_data, 64, args.mode)
    train_sample_weights = train_dataset.countweights()
    train_sampler = WeightedRandomSampler(train_sample_weights, len(train_sample_weights))
    train_dataloader = DataLoader(train_dataset, sampler = train_sampler, batch_size = args.train_batch_size)
    # train_dataloader = DataLoader(train_dataset, sampler=ImbalancedDatasetSampler(train_dataset, callback_get_label=lambda x, idx:x.__getlabel__(idx)), batch_size = args.train_batch_size)
    # train_dataloader = DataLoader(train_dataset, batch_size = args.train_batch_size, shuffle = True)

    eval_dataset = Mydata(eval_data, 32, args.mode)
    eval_sample_weights = eval_dataset.countweights()
    eval_sampler = WeightedRandomSampler(eval_sample_weights, len(eval_sample_weights))
    eval_dataloader = DataLoader(eval_dataset, sampler = eval_sampler, batch_size = args.eval_batch_size)
    # eval_dataloader = DataLoader(eval_dataset, sampler=ImbalancedDatasetSampler(eval_dataset, callback_get_label=lambda x, idx:x.__getlabel__(idx)), batch_size = args.eval_batch_size)
    # eval_dataloader = DataLoader(eval_dataset, batch_size = args.eval_batch_size, shuffle = True)

    criterion = nn.BCELoss()
    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)  

    total_train_losses, total_train_accs = [], []
    total_eval_losses, total_eval_accs = [], []
    eval_f1 = []
    best_acc = 0.0

    for epoch in range(1, args.num_epochs+1):

        print()
        print('epoch:{:d}/{:d}'.format(epoch, args.num_epochs))
        print('*' * 100)

        print("Training")
        epoch_train_loss, epoch_train_acc = train(model, train_dataloader, optimizer, criterion, device)
        epoch_train_loss = np.array(epoch_train_loss).mean()
        epoch_train_acc = np.array(epoch_train_acc).mean()
        print("Average loss:{:.4f}".format(epoch_train_loss))
        print("Average acc:{:.4f}".format(epoch_train_acc))
        total_train_losses.append(epoch_train_loss)
        total_train_accs.append(epoch_train_acc)

        print("Evaluating")
        epoch_eval_loss, epoch_eval_acc, f1 = evaluate(model, eval_dataloader, criterion, device)
        epoch_eval_loss = np.array(epoch_eval_loss).mean()
        epoch_eval_acc = np.array(epoch_eval_acc).mean()
        print("Average loss:{:.4f}".format(epoch_eval_loss))
        print("Average acc:{:.4f}".format(epoch_eval_acc))
        print("Average f1:{:.4f}".format(f1))
        total_eval_losses.append(epoch_eval_loss)
        total_eval_accs.append(epoch_eval_acc)
        eval_f1.append(f1)

        if epoch_eval_acc > best_acc:
            best_acc = epoch_eval_acc
            best_model = model
            # torch.save(best_model, args.model_dir+'/'+args.mode+'/best_model_{}.pt'.format(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())))
            torch.save(best_model, args.model_dir+'/'+args.mode+'/best_model.pt')
    
    dire = './image/simulator/'+args.mode
    x = range(args.num_epochs)
    plt.figure()
    plt.title('train and valid loss of {}-wise model'.format(args.mode))
    plt.xlabel(u'epochs')
    plt.ylabel(u'train and valid loss')
    plt.plot(x, total_train_losses, label='training loss')
    plt.plot(x, total_eval_losses, label='validation loss')
    plt.legend()
    plt.savefig(dire+'/train and valid loss curves of {}-wise model {}'.format(args.mode, time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())))

    plt.figure()
    plt.title('train and valid acc of {}-wise model'.format(args.mode))
    plt.xlabel(u'epochs')
    plt.ylabel(u'train and valid acc')
    plt.plot(x, total_train_accs, label='training acc')
    plt.scatter(total_train_accs.index(max(total_train_accs)), max(total_train_accs), label='best training acc = {:.4f}'.format(max(total_train_accs)))
    plt.plot(x, total_eval_accs, label='validation acc')
    plt.scatter(total_eval_accs.index(max(total_eval_accs)), max(total_eval_accs), label='best validation acc = {:.4f}'.format(max(total_eval_accs)))
    plt.legend()
    plt.savefig(dire+'/train and valid acc curves of {}-wise model {}'.format(args.mode, time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())))

    plt.figure()
    plt.title('f1 of {}-wise model'.format(args.mode))
    plt.xlabel(u'epochs')
    plt.ylabel(u'valid f1')
    plt.plot(x, eval_f1, label='validation f1')
    plt.legend()
    plt.savefig(dire+'/valid f1 curves of {}-wise model {}'.format(args.mode, time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())))
