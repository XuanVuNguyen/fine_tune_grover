import torch
from torch import nn
import numpy as np

from typing import Union
from collections import namedtuple, OrderedDict
import csv
import time
from itertools import chain
import json
import os
import glob

from grover.data.moldataset import MoleculeDatapoint, MoleculeDataset
from GROVER_finetune.model import GROVERFinetune
from GROVER_finetune.arguments import FinetuneArgs, DataArgs

def global_initiate():
    cuda = torch.cuda.is_available()
    if cuda:
        print('Cuda is available')
        print('Device name: ', torch.cuda.get_device_name())
        print('Device index: ', torch.cuda.current_device())
    else:
        print('Cuda is not available')
    return cuda

def initialize_model(model, 
                       pretrained_state_dict: Union[None, OrderedDict]):
    for layer_name, weight in model.named_parameters():
        if weight.dim() == 1:
            if not'act_func' in layer_name and not 'norm' in layer_name:
                nn.init.constant_(weight, 0.)
                print(f'Zero initialization for layer: {layer_name}')
        else:
            nn.init.xavier_normal_(weight)
            print(f'Xavier norm initialization for layer: {layer_name}')
            
    if pretrained_state_dict is not None:
        model.load_state_dict(pretrained_state_dict, strict=False)
        print('Pretrained weight loaded')
        
def get_pretrained_model(finetune_args, cuda):
    device = 'cuda' if cuda else 'cpu'
    model_pretrained = torch.load(finetune_args.pretrained_model_path, map_location=device)
    state_dict = model_pretrained['state_dict']
    args = model_pretrained['args']
    setattr(args, 'dropout', 0.0)
    setattr(args, 'cuda', cuda)
    return state_dict, args

def build_finetune_model(finetune_args, 
                         cuda) -> nn.Module:
    state_dict_pretrained, grover_args = get_pretrained_model(finetune_args, cuda)
    if not finetune_args.load_pretrained:
        state_dict_pretrained = None
    model = GROVERFinetune(finetune_args, grover_args)
    if cuda:
        model.cuda()
    initialize_model(model, state_dict_pretrained)
    return model


def get_optimizer(model, training_args):
    # base_layers = chain(model.grover.parameters(),
    #                     model.readout.parameters())
    # ffn_layers = chain(model.mol_atom_from_atom_ffn.parameters(), 
    #                    model.mol_atom_from_bond_ffn.parameters())
    # optimizer = torch.optim.Adam([{'params': base_layers, 
    #                                'lr': training_args.init_lr*training_args.finetune_coff},
    #                              {'params': ffn_layers, 'lr': training_args.init_lr}],
    #                              lr=training_args.init_lr,
    #                              weight_decay=training_args.weight_decay)
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=training_args.init_lr,
                                 weight_decay=training_args.weight_decay)
    return optimizer

def get_dataset(data_args):
    with open(data_args.data_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        lines = [line for line in csv_reader]
    if data_args.mol_feature_path:
        with np.load(data_args.mol_feature_path) as feature_file:
            features = feature_file['features']
        assert features.shape[0] == len(lines), "Number of molecules and number of feature sets mismatch"
        mol_datapoints = [MoleculeDatapoint(line, features=feature) for line, feature in zip(lines, features)]
    else:
        mol_datapoints = [MoleculeDatapoint(line) for line in lines]
    mol_dataset = MoleculeDataset(mol_datapoints)
    return mol_dataset
               
def create_save_dir(save_dir: str, time_prefix: str):
    if time_prefix:
        if save_dir[-1]=='/':
            save_dir, _, _ = save_dir.rpartition('/')
        mother_dir, _, child_dir = save_dir.rpartition('/')
        child_dir = time.strftime('%Y_%m_%d-%H_%M_%S-') + child_dir
        save_dir = os.path.join(mother_dir, child_dir)
    
    weight_dir = os.path.join(save_dir, 'weight')
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
        
    return save_dir

def save_args(save_dir: str, **argument_sets):
    for key, arg_set in argument_sets.items():
        if not isinstance(arg_set, dict):
            argument_sets[key]=arg_set._asdict()
    with open(os.path.join(save_dir, 'arguments.txt'), 'w') as file:
        json.dump(argument_sets, file, indent=4)
        
def load_model_from_checkpoint(finetune_args, 
                               checkpoint_path: str,
                               cuda):
    _, grover_args = get_pretrained_model(finetune_args, cuda)
    model = GROVERFinetune(finetune_args, grover_args)
    device = 'cuda' if cuda else 'cpu'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    if cuda:
        model.cuda()
    return model


def unpack_checkpoint_dir(evaluation_args, cuda):
    with open(os.path.join(evaluation_args.checkpoint_dir, 'arguments.txt'), 'r') as file:
        arguments = json.load(file)
        data_args_dict, finetune_args_dict = arguments['data_args'], arguments['finetune_args']
    data_args, finetune_args = DataArgs(**data_args_dict), FinetuneArgs(**finetune_args_dict)
    checkpoint_path = glob.glob(os.path.join(evaluation_args.checkpoint_dir, 'weight', 'epoch-{:03}-*'.format(evaluation_args.epoch)))
    model = load_model_from_checkpoint(finetune_args, *checkpoint_path, cuda)
    return model, data_args
