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

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset

def train_fl_gnn(args, data_config):
    NUM_CLIENTS = 10

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    


    def set_parameters(net, parameters: List[np.ndarray]):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        #net.load_state_dict(state_dict, strict=False)


    def get_parameters(net) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in net.state_dict().items()]
    
    class FlowerClient(NumPyClient):
        def __init__(self, net, trainloader, valloader,testloader, tr_inds, val_inds, te_inds, optimizer,loss_fn, args, wandb_config, device, val_data, te_data, data_config):
            self.net = net
            self.trainloader = trainloader
            self.valloader = valloader
            self.testloader = testloader
            self.tr_inds = tr_inds
            self.val_inds = val_inds
            self.te_inds = te_inds
            self.optimizer = optimizer
            self.loss_fn = loss_fn
            self.args = args
            self.wandb_config = wandb_config
            self.device = device
            self.val_data = val_data
            self.te_data = te_data
            self.data_config = data_config

        def get_parameters(self, config):
            return get_parameters(self.net)

        def fit(self, parameters, config):
            set_parameters(self.net, parameters)

            if args.reverse_mp:
                train_fl_hetero(self.trainloader, self.valloader, self.testloader, self.tr_inds, self.val_inds, self.te_inds, self.net, self.optimizer, self.loss_fn, self.args,self.wandb_config,self.device, self.val_data, self.te_data, self.data_config)
                
            else:
                train_fl_homo(self.trainloader, self.valloader, self.testloader, self.tr_inds, self.val_inds, self.te_inds, self.net, self.optimizer, self.loss_fn, self.args,self.wandb_config,self.device, self.val_data, self.te_data, self.data_config)

            return get_parameters(self.net), len(self.trainloader), {}

        def evaluate(self, parameters, config):
            set_parameters(self.net, parameters)
            if args.reverse_mp:
                f1 = evaluate_hetero(self.testloader, self.te_inds,self.net, self.te_data, self.device, self.args) #f1'nı dene
            else:
                f1 = evaluate_homo(self.testloader, self.te_inds,self.net, self.te_data, self.device, self.args)
            
            return len(self.valloader), {"f1": float(f1)}


    def client_fn(context: Context) -> Client:
        """Create a Flower client representing a single organization."""
        # Load data 
        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data partition
        # Read the node_config to fetch data partition associated to this node

        partition_id = context.node_config["partition-id"]

        tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)

        if args.ego:
            transform = AddEgoIds()
        else:
            transform = None

        #add the unique ids to later find the seed edges
        add_arange_ids([tr_data, val_data, te_data])

        tr_loader, val_loader, te_loader = get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args)

        sample_batch = next(iter(tr_loader))
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
        # Create a single Flower client representing a single organization
        # FlowerClient is a subclass of NumPyClient, so we need to call .to_client()
        # to convert it to a subclass of `flwr.client.Client`
        trainloader, valloader, testloader = get_fl_loaders(partition_id, args, data_config)

        tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_fl_data(partition_id, args, data_config)

        return FlowerClient(net, trainloader, valloader, testloader, tr_inds, val_inds, te_inds, optimizer, loss_fn, args, wandb_config, DEVICE, val_data,te_data, data_config ).to_client()


    # Create the ClientApp
    client = ClientApp(client_fn=client_fn)

    def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
        f1 = [num_examples * m["f1"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        # Aggregate and return custom metric (weighted average)
        return {"f1": sum(f1) / sum(examples)}
    
    # Specify the resources each of your clients need
    # By default, each client will be allocated 1x CPU and 0x GPUs
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

    # When running on GPU, assign an entire GPU for each client
    if DEVICE.type == "cuda":
        backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}
        # Refer to our Flower framework documentation for more details about Flower simulations
        # and how to set up the `backend_config`
    
    def server_fn(context: Context) -> ServerAppComponents:
        """Construct components that set the ServerApp behaviour.

        You can use settings in `context.run_config` to parameterize the
        construction of all elements (e.g the strategy or the number of rounds)
        wrapped in the returned ServerAppComponents object.
        """

        # Create FedAvg strategy
        strategy = FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=0.5,
            min_fit_clients=10,
            min_evaluate_clients=5,
            min_available_clients=10,
            evaluate_metrics_aggregation_fn=weighted_average,  
        )

        # Configure the server for 5 rounds of training
        config = ServerConfig(num_rounds=5)

        return ServerAppComponents(strategy=strategy, config=config)


    # Create a new server instance with the updated FedAvg strategy
    server = ServerApp(server_fn=server_fn)

    # Run simulation
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_CLIENTS,
        backend_config=backend_config,
    )



def train_fl_homo(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data,te_data,  data_config):
    #training
    best_val_f1 = 0
    for epoch in range(config.epochs):
        total_loss = total_examples = 0
        preds = []
        ground_truths = []
        for batch in tqdm.tqdm(tr_loader, disable=not args.tqdm):
            optimizer.zero_grad()
            #select the seed edges from which the batch was created
            inds = tr_inds.detach().cpu()
            batch_edge_inds = inds[batch.input_id.detach().cpu()]
            batch_edge_ids = tr_loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
            mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

            #remove the unique edge id from the edge features, as it's no longer needed
            batch.edge_attr = batch.edge_attr[:, 1:]

            batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            pred = out[mask]
            ground_truth = batch.y[mask]
            preds.append(pred.argmax(dim=-1))
            ground_truths.append(ground_truth)
            loss = loss_fn(pred, ground_truth)

            loss.backward()
            optimizer.step()

            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()

        pred = torch.cat(preds, dim=0).detach().cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).detach().cpu().numpy()
        f1 = f1_score(ground_truth, pred)
        print(f'Train F1: {f1:.4f}')

        #evaluate
        val_f1 = evaluate_homo(val_loader, val_inds, model, val_data, device, args)
        te_f1 = evaluate_homo(te_loader, te_inds, model, te_data, device, args)

        print(f'Validation F1: {val_f1:.4f}')
        print(f'Test F1: {te_f1:.4f}')

        if epoch == 0:
            continue
        elif val_f1 > best_val_f1:
            best_val_f1 = val_f1
            if args.save_model:
                save_model(model, optimizer, epoch, args, data_config)
    
    return model

def train_fl_hetero(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data,te_data,  data_config):
    #training
    best_val_f1 = 0
    for epoch in range(config.epochs):
        total_loss = total_examples = 0
        preds = []
        ground_truths = []
        for batch in tqdm.tqdm(tr_loader, disable=not args.tqdm):
            optimizer.zero_grad()
            #select the seed edges from which the batch was created
            inds = tr_inds.detach().cpu()
            batch_edge_inds = inds[batch['node', 'to', 'node'].input_id.detach().cpu()]
            batch_edge_ids = tr_loader.data['node', 'to', 'node'].edge_attr.detach().cpu()[batch_edge_inds, 0]
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
        print(f'Train F1: {f1:.4f}')

        #evaluate
        val_f1 = evaluate_hetero(val_loader, val_inds, model, val_data, device, args)
        te_f1 = evaluate_hetero(te_loader, te_inds, model, te_data, device, args)

        
        print(f'Validation F1: {val_f1:.4f}')
        print(f'Test F1: {te_f1:.4f}')

        if epoch == 0:
            continue
        elif val_f1 > best_val_f1:
            best_val_f1 = val_f1
            if args.save_model:
                save_model(model, optimizer, epoch, args, data_config)
        