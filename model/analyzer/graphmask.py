import math
import sys
import os

import numpy as np
import pandas as pd
import torch
from torch.nn import ReLU, Linear
from torch.optim.lr_scheduler import StepLR
from utils.hard_concrete import HardConcrete
from utils.multiple_inputs_layernorm_linear import MultipleInputsLayernormLinear
from utils.squeezer import Squeezer
from utils.moving_average import MovingAverage
from utils.path import Path
from utils.lagrangian_optimization import LagrangianOptimization
from model.trip_popularity import MyHetero
from model.trip_popularity import get_data
from collections import defaultdict
import yaml
import wandb


def calc_cor(data, prediction):
    mask = data['spot']['test_mask']
    cor = np.corrcoef(data['spot'].y[mask].cpu().detach().numpy(), prediction[mask].squeeze().cpu().detach().numpy())[0][1]
    return cor
    
    

class GraphMaskProbe(torch.nn.Module):

    device = None

    def __init__(self, vertex_embedding_dims, message_dims, h_dims, edge_type):
        super().__init__()
        gates = []
        baselines = []
        for v_dim, m_dim, h_dim in zip(vertex_embedding_dims, message_dims, h_dims):
            gate_input_shape = [v_dim, m_dim, v_dim]
            gate = torch.nn.Sequential(
                MultipleInputsLayernormLinear(gate_input_shape, h_dim),
                ReLU(),
                Linear(h_dim, 1),
                Squeezer(),
                HardConcrete()
            )

            gates.append(gate)

            baseline = torch.FloatTensor(m_dim)
            stdv = 1. / math.sqrt(m_dim)
            baseline.uniform_(-stdv, stdv)
            baseline = torch.nn.Parameter(baseline, requires_grad=True)

            baselines.append(baseline)

        gates = torch.nn.ModuleList(gates)
        self.gates = gates

        baselines = torch.nn.ParameterList(baselines)
        self.baselines = baselines

        # Initially we cannot update any parameters. They should be enabled layerwise
        for parameter in self.parameters():
            parameter.requires_grad = False

    def enable_layer(self, layer):
        print("Enabling layer "+str(layer), file=sys.stderr)
        for parameter in self.gates[layer].parameters():
            parameter.requires_grad = True

        self.baselines[layer].requires_grad = True


    def forward(self, model, edge_type):
        latest_source_embeddings = model.get_latest_source_embeddings(edge_type)
        latest_messages = model.get_latest_messages(edge_type)
        latest_target_embeddings = model.get_latest_target_embeddings(edge_type)

        gates = []
        total_penalty = 0
        for i in range(len(self.gates)):
            gate_input = [latest_source_embeddings[i], latest_messages[i], latest_target_embeddings[i]]
            gate, penalty = self.gates[i](gate_input)

            gates.append(gate)
            total_penalty += penalty

        return gates, self.baselines, total_penalty

    def save(self, path):
        print("Saving to path " + path, file=sys.stderr)
        torch.save(self.state_dict(), path)

    def load(self, path):
        print("Loading from path " + path, file=sys.stderr)

        self.load_state_dict(torch.load(path, map_location=self.device))

    def set_device(self, device):
        self.device = device
        self.to(device)

class GraphMaskAnalyser:

    probe = None
    moving_average_window_size = 100

    def __init__(self, configuration, edge_type_list):
        self.configuration = configuration
        self.edge_type_list = edge_type_list
    #TODO:全てproblemを消す
    def initialise_for_model(self, model):
        wandb.init(project='graph mask',config=self.configuration)
        self.probe_dict = torch.nn.ModuleDict()
        for edge_type in self.edge_type_list:
            vertex_embedding_dims = model.get_vertex_embedding_dims(edge_type)
            message_dims = model.get_message_dims(edge_type)
            self.probe_dict[edge_type] = GraphMaskProbe(vertex_embedding_dims, message_dims, message_dims, edge_type)

    
        '''
        FIXME: 
        else:
            vertex_embedding_dims = model.get_gnn().get_vertex_embedding_dims()
            message_dims = model.get_gnn().get_message_dims()
            n_relations = model.get_gnn().n_relations
            self.probe = GraphMaskAdjMatProbe(vertex_embedding_dims, message_dims, n_relations, vertex_embedding_dims)
        '''
    def fit(self, model, data, gpu_number=-1):
        batch_size = self.configuration["analysis"]["parameters"]["batch_size"]
        epochs_per_layer = self.configuration["analysis"]["parameters"]["epochs_per_layer"]
        train_split = self.configuration["analysis"]["parameters"]["train_split"]
        test_every_n = self.configuration["analysis"]["parameters"]["test_every_n"]
        save_path = self.configuration["analysis"]["parameters"]["save_path"]
        penalty_scaling = self.configuration["analysis"]["parameters"]["penalty_scaling"]
        learning_rate = self.configuration["analysis"]["parameters"]["learning_rate"]
        allowance = self.configuration["analysis"]["parameters"]["allowance"]
        max_allowed_performance_diff = self.configuration["analysis"]["parameters"]["max_allowed_performance_diff"]
        load = self.configuration["analysis"]["parameters"]["load"]
        train = self.configuration["analysis"]["parameters"]["train"]
        max_allowed_performance_diffs = [0.05, 0.03, 0.02]

        if load:
            self.probe.load(save_path)

        if train:
            best_sparsity = 1.01
            for edge_type in self.edge_type_list:
                probe = self.probe_dict[edge_type]
                probe.save(save_path)
                optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
                scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
                lagrangian_optimization = LagrangianOptimization(optimizer, device, batch_size_multiplier=None)

                for layer in reversed(list(range(len(model.get_gnn())))): 
                    probe.enable_layer(layer)

                    for epoch in range(epochs_per_layer):
                        probe.train()
                        loss, predictions, penalty = self.compute_graphmask_loss(data, model, probe, edge_type)
                        print('epoch', epoch, 'loss',loss,'penalty',penalty)

                        g = torch.relu(loss-allowance).mean()
                        f = penalty*penalty_scaling

                        lagrangian_optimization.update(f, g)
                        scheduler.step()

                        if (epoch+1)%test_every_n==0:
                            percent_div, sparsity = self.validate(data, model, probe, edge_type)
                            print('epoch', epoch, 'percent_div', percent_div, 'sparsity', sparsity)
                            wandb.log({'epoch':epoch, 'percent_div':percent_div})
                            wandb.log({'epoch':epoch, 'sparsity':sparsity})


                            if percent_div< max_allowed_performance_diffs[layer] and sparsity< best_sparsity:
                                print("Found better probe with sparsity={0:.4f}. Keeping these parameters.".format(sparsity), file=sys.stderr)
                                best_sparsity = sparsity
                                probe.save(save_path)

                        probe.load(save_path)

    def validate(self, data, model, probe, edge_type, split='test'):
        model.eval()
        #model.reset_injected_messages()
        _, original_predictions = model(data, replacement=False)
        original_score = calc_cor(data, original_predictions)

        model.overwrite_label(original_predictions)


        probe.eval()
        gates, baselines, penalty = probe(model, edge_type)

        model.inject_message_scale(gates, edge_type)
        model.inject_message_replacement(baselines, edge_type)

        all_messages = sum([len(gate) for gate in gates])
        all_gates = sum([sum(gate) for gate in gates])
        sparsity = float(all_gates/all_messages)
        _, gated_predictions = model(data, replacement=True)
        gated_score = calc_cor(data, gated_predictions)
        diff = original_score-gated_score
        percent_div = diff/original_score
        print('original score:',original_score, 'gated_score:',gated_score)

        return percent_div, sparsity

    def validate_(self, data, model, probe, edge_type,split="test", gpu_number=-1):
        probe.to(device)

        model.to(device)
        data.to(device)

        model.reset_injected_messages()
        model.eval()
        _, original_predictions = model(data)
        original_score = calc_cor(data, original_predictions)

        print('validate original_score',original_score)
        probe.eval()
        
        gates, baselines, _ = probe(model, edge_type)
        print('valid gates', gates, 'valid baseines', baselines)
        all_edge = len(gates[0])+len(gates[1])
        remain_edge = len(gates[0][gates[0]==0])+len(gates[1][gates[1]==0])
        print('validate sparsity', remain_edge/all_edge)
        model.inject_message_scale(gates, edge_type)
        model.inject_message_replacement(baselines, edge_type)
        _, predictions = model(data)
        gated_score = calc_cor(data, predictions)
        print('validate gated_score',gated_score)
        diff = np.abs(original_score - gated_score)
        percent_div = float(diff / (original_score + 1e-8))

        all_gates=sum([len(gate) for gate in gates])
        all_messages = sum([sum(gate) for gate in gates])
        sparsity = float(all_gates / all_messages)

        return percent_div, sparsity

    def compute_graphmask_loss(self, data, model, probe, edge_type):

        model.eval()
        _, original_predictions = model(data, replacement=False)

        model.overwrite_label(original_predictions)


        model.train()
        probe.train()
        probe.to(device)

        gates, baselines, penalty = probe(model, edge_type)
        model.inject_message_scale(gates, edge_type)
        model.inject_message_replacement(baselines, edge_type)
        print('compute graph mask loss', model.injected_message_scale_dict)
        all_edge = len(gates[0])+len(gates[1])+len(gates[2])
        remain_edge = len(gates[0][gates[0]==0])+len(gates[1][gates[1]==0])+len(gates[1][gates[1]==0])
        print('sparsity', remain_edge/all_edge)
        loss, predictions = model(data, replacement=True)

        return loss, predictions, penalty





    def compute_graphmask_loss_(self, batch, model, problem):
        model.eval()
        _, original_predictions = model(batch, replacement=False)

        model.train() # Enable any dropouts in the original model. We found this helpful for training GraphMask.
        self.probe.train()

        #TODO: これはモデルに保存する
        batch = problem.overwrite_labels(batch, original_predictions)

        gates, baselines, penalty = self.probe(model.get_gnn())
        model.get_gnn().inject_message_scale(gates)
        model.get_gnn().inject_message_replacement(baselines)

        #TODO: 上で保存した値との差分を計算する関数を作る
        loss, predictions = model(batch, replacement=True)

        return loss, predictions, penalty

    def analyse(self, batch, model, problem):
        model.eval()
        self.probe.eval()
        _, original_predictions = model(batch)

        gates, _, _ = self.probe(model.get_gnn())

        return gates

if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    with open('/home/yamanishi/project/trip_recommend/analyzer/configuration/trip_popularity.yaml') as file:
        config = yaml.safe_load(file)
    path = Path()
    df = pd.read_csv(path.df_experience_light_path)
    data = get_data(df,config)
    print(data)
    model = MyHetero(config, data)
    model.load_state_dict(torch.load('/home/yamanishi/project/trip_recommend/data/model/model_new.pth'))
    data.to(device)
    model.to(device)   
    torch.autograd.set_detect_anomaly(True)
    analyzer = GraphMaskAnalyser(configuration=config, edge_type_list=['word__revrelate__spot'])
    analyzer.initialise_for_model(model)
    analyzer.fit(model, data, gpu_number=5)

    #print(analyzer.probe_dict)
