from collections import OrderedDict
from typing import List, Tuple
import numpy as np
import logging
import tqdm

import torch
from train_util import add_arange_ids, AddEgoIds, get_loaders, extract_param, save_model, evaluate_hetero, evaluate_homo
from training import get_model
from train_fl_util import get_fl_loaders
from data_loading import get_fl_data
from torch_geometric.nn import to_hetero, summary
from sklearn.metrics import f1_score
from torch_geometric.data import Data, HeteroData
from data_loading import get_data
from models import GINe
from util import logger_setup
logger_setup()
from sklearn.metrics import precision_score, recall_score

import json
from util import create_parser

parser = create_parser()
args = parser.parse_args()

with open('data_config.json', 'r') as config_file:
    data_config = json.load(config_file)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_ind_hetero(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data,te_data,  data_config):
    #training
    best_val_f1 = 0
    model.train()
    
    for epoch in range(config.epochs):
        total_loss = total_examples = 0
        preds = []
        ground_truths = []
        for batch in tqdm.tqdm(tr_loader, disable=not args.tqdm):
            optimizer.zero_grad()
            # Get the indices for the current batch
            inds = tr_inds.detach().cpu()
            batch_edge_inds = inds[batch['node', 'to', 'node'].input_id.detach().cpu()]
            
            # Important: Ensure indices are within bounds
            batch_edge_inds = batch_edge_inds[batch_edge_inds < tr_loader.data['node', 'to', 'node'].edge_attr.size(0)]
            
            # Get edge IDs safely
            batch_edge_ids = tr_loader.data['node', 'to', 'node'].edge_attr[batch_edge_inds, 0]
            mask = torch.isin(batch['node', 'to', 'node'].edge_attr[:, 0].detach().cpu(), batch_edge_ids)
            
            #remove the unique edge id from the edge features, as it's no longer needed
            batch['node', 'to', 'node'].edge_attr = batch['node', 'to', 'node'].edge_attr[:, 1:]
            batch['node', 'rev_to', 'node'].edge_attr = batch['node', 'rev_to', 'node'].edge_attr[:, 1:]

            
            batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
            out = out[('node', 'to', 'node')]
            pred = out[mask]
            ground_truth = batch['node', 'to', 'node'].y[mask]
            preds.append(pred.argmax(dim=-1))
            ground_truths.append(batch['node', 'to', 'node'].y[mask])
            loss = loss_fn(pred, ground_truth)

            loss.backward()
            optimizer.step()

            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
            
        pred = torch.cat(preds, dim=0).detach().cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).detach().cpu().numpy()
        f1 = f1_score(ground_truth, pred)
        #calculate precision, recall

        precision = precision_score(ground_truth, pred, average='binary')
        recall = recall_score(ground_truth, pred, average='binary') 
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")

        print(f"Training Loss:", total_loss/total_examples)
        print(f'Train F1: {f1:.4f}')


config={
            "epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "model": args.model,
            "data": args.data,
            "num_neighbors": args.num_neighs,
            "lr": extract_param("lr", args),
            "n_hidden": extract_param("n_hidden", args),
            "n_gnn_layers": extract_param("n_gnn_layers", args),
            "loss": "ce",
            "w_ce1": extract_param("w_ce1", args),
            "w_ce2": extract_param("w_ce2", args),
            "dropout": extract_param("dropout", args),
            "final_dropout": extract_param("final_dropout", args),
            "n_heads": extract_param("n_heads", args) if args.model == 'gat' else None
        }
    
class DictToObj:
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)
    
wandb_config = DictToObj(config)


partition_id = args.partition_id

print("Running on partition", partition_id)
print("Params:", config)

trainloader, valloader, testloader, tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_fl_loaders(partition_id, args, data_config)

sample_batch = next(iter(trainloader))
sample_batch.to(DEVICE)

net = get_model(sample_batch, wandb_config, args)

if args.reverse_mp:
    net = to_hetero(net, te_data.metadata(), aggr='mean')

net.to(DEVICE)
optimizer = torch.optim.Adam(net.parameters(), lr=wandb_config.lr)

sample_x = sample_batch.x if not isinstance(sample_batch, HeteroData) else sample_batch.x_dict
sample_edge_index = sample_batch.edge_index if not isinstance(sample_batch, HeteroData) else sample_batch.edge_index_dict

if isinstance(sample_batch, HeteroData):
    sample_batch['node', 'to', 'node'].edge_attr = sample_batch['node', 'to', 'node'].edge_attr[:, 1:]
    sample_batch['node', 'rev_to', 'node'].edge_attr = sample_batch['node', 'rev_to', 'node'].edge_attr[:, 1:]
else:
    sample_batch.edge_attr = sample_batch.edge_attr[:, 1:]

sample_edge_attr = sample_batch.edge_attr if not isinstance(sample_batch, HeteroData) else sample_batch.edge_attr_dict

print(summary(net, sample_x, sample_edge_index, sample_edge_attr))

loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([wandb_config.w_ce1, wandb_config.w_ce2]).to(DEVICE))

train_ind_hetero(trainloader, valloader, testloader, tr_inds, val_inds, te_inds, net, optimizer, loss_fn, args, wandb_config,DEVICE, val_data, te_data, data_config)

_ = evaluate_hetero(testloader, te_inds, net, te_data, DEVICE, args)