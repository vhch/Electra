
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm

from transformers import AutoConfig, get_cosine_with_hard_restarts_schedule_with_warmup
from transformers import ElectraForSequenceClassification, ElectraForPreTraining, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from torchmetrics import Accuracy



def metric(model, valid_loader):
    preds, target = torch.tensor([]), torch.tensor([])
    model.eval()
    for iter, (x, y, z) in enumerate(tqdm(valid_loader)):
        x = x.to(device=device)
        y = y.to(device=device)
        out = model(input_ids = x, attention_mask = y).logits.detach().cpu()
        preds = torch.cat([preds, out], dim=0)
        target = torch.cat([target, z], dim=0)

    accuracy = Accuracy(top_k=1)
    accuracy2 = Accuracy(top_k=2)
    accuracy3 = Accuracy(top_k=3)
    target = target.type(torch.int64)
    return accuracy(preds, target), accuracy2(preds, target), accuracy3(preds, target)

def loss_func(out, y):
    return F.cross_entropy(out, y)


if __name__ == '__main__':
    
    args = argparse.ArgumentParser()

    args.add_argument('--mode', type=str, default='train')
    
    config = args.parse_args()

    # Hyperparameters & Settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = MyModel().to(device=device)

    # my_config = AutoConfig.from_pretrained('albert-base-v2', max_position_embeddings = 101, vocab_size = VOCAB_SIZE,
    #  num_labels=CLASS_NUM)
    # my_config = AutoConfig.from_pretrained('prajjwal1/bert-tiny', num_labels=CLASS_NUM)
    my_config = AutoConfig.from_pretrained('google/electra-base-discriminator', num_labels=5, vocab_size = 10000, max_position_embeddings = 512)


    # model_load = ElectraForPreTraining(my_config)
    # model_load = model_load.to(device=device)


    model = ElectraForSequenceClassification(my_config)
    model = model.to(device=device)

    print(model)


###################################################################################################################################
    # Load data
    print('Load data...')
    train_dataset_path = 'DATASET_PATH'
    train_x = np.load(train_dataset_path + '/train_data/x.npy').astype(np.int64)
    train_y = np.load(train_dataset_path + '/train_label').astype(np.int64)
    # train_dataset = MyDataset(train_x, train_y)


    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=49)

    for train_index, test_index in skf.split(train_x, train_y):
        x_train, x_test = train_x[train_index], train_x[test_index]
        y_train, y_test = train_y[train_index], train_y[test_index]
        break

    attention_mask = []
    for i in x_train:
        temp = []
        for j in i:
            if j == 0:
                temp.append(0)
            else:
                temp.append(1)
        attention_mask.append(temp)

    attention_mask_test = []
    for i in x_test:
        temp = []
        for j in i:
            if j == 0:
                temp.append(0)
            else:
                temp.append(1)
        attention_mask_test.append(temp)


    def masked(all_x):
        rand = torch.rand(all_x.shape)
        mask_arr = (rand < 0.15) * torch.tensor(all_x != 0) * torch.tensor(all_x != 4998) 

        selection = []
        for i in range(all_x.shape[0]):
            selection.append(
                torch.flatten(mask_arr[i].nonzero())
            )
        for i in range(all_x.shape[0]):
            selection_val = np.random.random(len(selection[i])) # selection의 위치마다 0~1 값 부여
            mask_selection = selection[i][np.where(selection_val >= 0.2)[0]] # 80% : Mask 토큰 대체
            random_selection = selection[i][np.where(selection_val < 0.1)[0]] # 10% : 랜덤 토큰 대체
            all_x[i, mask_selection] = 10000
            all_x[i, random_selection] = torch.randint(0, 10000, size = random_selection.shape)
        return all_x


    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(attention_mask), torch.tensor(y_train))
    valid_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(attention_mask_test), torch.tensor(y_test))

    print('Loading data is finished!')
###################################################################################################################################
    # Hyperparameters 
    epoch_train = 100
    batch_size = 32

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size = batch_size)
    valid_loader = DataLoader(valid_dataset, shuffle=True, batch_size = batch_size)

    # save layer names
    layer_names = []
    for idx, (name, param) in enumerate(model.named_parameters()):
        layer_names.append(name)
        # print(f'{idx}: {name}')
    
    layer_names.reverse()

    lr = 1e-5
    lr_mult = 1
    # lr_mult = 0.98

    # placeholder
    parameters = []

    # store params & learning rates
    for idx, name in enumerate(layer_names):
        
        # display info
        print(f'{idx}: lr = {lr}, {name}')
        
        # append layer parameters
        parameters += [{'params': [p for n, p in model.named_parameters() if n == name and p.requires_grad],
                        'lr': lr, 'weight_decay':0}]
        
        # update learning rate
        lr *= lr_mult

    # Optimizers
    # optim = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay = 0.01)
    # optim = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay = 0)
    optim = torch.optim.AdamW(parameters)
    # scheduler = get_linear_schedule_with_warmup(
    #                                             optimizer=optim,
    #                                             num_warmup_steps=0,
    #                                             num_training_steps=epoch_train * len(train_loader))
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                                                optimizer=optim,
                                                num_warmup_steps=0,
                                                num_training_steps=epoch_train * len(train_loader),
                                                num_cycles=1)



    # Train
    for epoch in range(epoch_train):
        model.train()
        cnt = 0
        loss_sum = 0
        # for x, y, z in train_loader:
        for iter, (x, y, z) in enumerate(tqdm(train_loader)):
            optim.zero_grad()
            x = x.to(device=device)
            y = y.to(device=device)
            z = z.to(device=device)
            out = model(input_ids = x, attention_mask = y, labels = z)
            loss = out[0]
            loss.backward()
            optim.step()
            scheduler.step()

            loss_sum += loss.item()
            cnt += 1
        # scheduler.step()
        loss_mean = loss_sum / cnt

        accuracy1, accuracy2, accuracy3 = metric(model, valid_loader)

        # Report loss
        # results = {'train_loss': loss_mean}
        results = {'train_loss': loss_mean, 'accuracy_tr': float(accuracy1), 'accuracy_val': float(accuracy1)}

        print(f'Save epoch = {epoch+1}, train_loss = {loss_mean}, a1 = {accuracy1}, a2 = {accuracy2}, a3 = {accuracy3}')

