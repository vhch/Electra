
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm

from transformers import AutoConfig, get_cosine_with_hard_restarts_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers import ElectraForPreTraining, ElectraForMaskedLM, ElectraForSequenceClassification


import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

import copy


class Electra(nn.Module):
    def __init__(self, generator, discriminator):
        super().__init__()

        self.generator = generator
        self.discriminator = discriminator

    def forward(self, x, y, label_g, label_d):
        output_g = self.generator(input_ids = x, attention_mask = y, labels = label_g)
        x = torch.argmax(output_g[1], dim = -1).detach()
        loss_g = output_g[0]
        output_d = self.discriminator(input_ids = x, attention_mask = y, labels = label_d)
        loss_d = output_d[0]
        loss = loss_g * 1.0 + loss_d * 50.0

        return loss, loss_g, loss_d

def get_params_without_weight_decay_ln(named_params, weight_decay):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay,
        },
        {
            'params': [p for n, p in named_params if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]
    return optimizer_grouped_parameters


def masked(x):
    rand = torch.rand(x.shape)
    # label = torch.zeros(x.shape)
    label = torch.zeros(x.shape)
    # mask_arr = (rand < 0.15) * torch.tensor(x != 0) * torch.tensor(x != 4998) 
    mask_arr = (rand < 0.15) * (x != 0) * (x != 4998)

    selection = []
    for i in range(x.shape[0]):
        selection.append(
            torch.flatten(mask_arr[i].nonzero())
        )
    for i in range(x.shape[0]):
        selection_val = np.random.random(len(selection[i])) # selection의 위치마다 0~1 값 부여
        mask_selection = selection[i][np.where(selection_val >= 0)[0]] # 
        x[i, mask_selection] = 4999
        label[i, mask_selection] = 1
    return x, label

def setup(rank, world_size):
    # os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8080'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(args, model, rank, world_size, train_loader, optimizer, epoch, scheduler, sampler=None):
    model.train()
    ddp_loss = torch.zeros(3).to(rank)
    if sampler:
        sampler.set_epoch(epoch)

    if rank==0:
        inner_pbar = tqdm(
            range(len(train_loader))
        )

    scaler = GradScaler()

    for iter, (x, y, label_g) in enumerate(train_loader):
        optimizer.zero_grad()
        x, label_d = masked(x)
        x, y, label_g, label_d = x.to(rank), y.to(rank), label_g.to(rank), label_d.to(rank)
        with autocast():
            loss, loss_g, loss_d = model(x, y, label_g, label_d)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ddp_loss[0] += loss_g.item()
        # ddp_loss[1] += len(x)
        ddp_loss[1] += loss_d.item()
        ddp_loss[2] += 1


        if rank==0: 
            inner_pbar.update(1)

    # dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0:
        inner_pbar.close()
        print(f'pre epoch = {epoch+1}, g loss = {ddp_loss[0] / ddp_loss[2]}, d loss = {ddp_loss[1] / ddp_loss[2]}')

def fsdp_main(rank, world_size, args):
    setup(rank, world_size)

    # Hyperparameters & Settings
    torch.cuda.set_device(rank)
    device = torch.device(rank)


    # Bind model
    config_g = AutoConfig.from_pretrained('google/electra-base-generator', num_labels=5, vocab_size = 10000, max_position_embeddings = 512)
    config_d = AutoConfig.from_pretrained('google/electra-base-discriminator', num_labels=5, vocab_size = 10000, max_position_embeddings = 512)

    model_g = ElectraForMaskedLM(config_g)
    model_g = model_g.to(rank)

    model_d = ElectraForPreTraining(config_d)
    model_d = model_d.to(rank)

    model_g.electra.embeddings.word_embeddings = model_d.electra.embeddings.word_embeddings
    model_g.electra.embeddings.position_embeddings = model_d.electra.embeddings.position_embeddings
    model_g.electra.embeddings.token_type_embeddings = model_d.electra.embeddings.token_type_embeddings

    model_save = ElectraForSequenceClassification(config_d)
    model_save.electra = None
    model_save = model_save.to(rank)
    model_save.electra = model_d.electra
    
    # model_d = DDP(model_d, device_ids=[rank], find_unused_parameters=True)
    model = Electra(model_g, model_d)
    model = DDP(model, device_ids=[rank])


    # Load data
    print('Load data...')
    train_dataset_path = 'DATASET_PATH'
    x = np.load(train_dataset_path).astype(np.int64)  # Pre-training data

    attention_mask_pre = []
    for i in x:
        temp = []
        for j in i:
            if j == 0:
                temp.append(0)
            else:
                temp.append(1)
        attention_mask_pre.append(temp)

    # label_g = x.copy()
    label_g = copy.deepcopy(x)

    # print(x[0], label_g[0], label_d[0])

    dataset = TensorDataset(torch.tensor(x), torch.tensor(attention_mask_pre), torch.tensor(label_g))
    sampler1 = DistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=True)
    train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1}
    cuda_kwargs = {'num_workers': 2,
                    'pin_memory': True,
                    'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    train_loader = torch.utils.data.DataLoader(dataset,**train_kwargs)

    print('Loading data is finished!')
    # print(len(train_loader))

    # pre Optimizers
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay = 0.01)
    optimizer = torch.optim.AdamW(get_params_without_weight_decay_ln(model.named_parameters(), 0.01), lr=2e-4, weight_decay = 0.01)
    # scheduler = get_linear_schedule_with_warmup(
    #                                             optimizer=optimizer,
    #                                             num_warmup_steps=10000,
    #                                             num_training_steps=args.epochs * len(train_loader))
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                                                optimizer=optimizer,
                                                num_warmup_steps=10000,
                                                num_training_steps=args.epochs * len(train_loader),
                                                num_cycles=1)

    for epoch in range(args.epochs):
        train(args, model, rank, world_size, train_loader, optimizer, epoch, scheduler, sampler = sampler1)
        # scheduler.step()

        # use a barrier to make sure training is done on all ranks
        dist.barrier()

        # if rank == 0:
        #     save(epoch+1)


    cleanup()

if __name__ == '__main__': 

    
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=400, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_main,
        args=(WORLD_SIZE, args),
        nprocs=WORLD_SIZE,
        join=True)