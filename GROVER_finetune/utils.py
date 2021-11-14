import torch
from torch import nn

from typing import Union
from collections import namedtuple, OrderedDict
import csv
import time
from itertools import chain
import json
import os

from grover.data.moldataset import MoleculeDatapoint, MoleculeDataset
from GROVER_finetune.model import GROVERFinetune

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
    for weight in model.parameters():
        if weight.dim() == 1:
            nn.init.constant_(weight, 0)
        else:
            nn.init.xavier_normal_(weight)
            
    if pretrained_state_dict is not None:
        model.load_state_dict(pretrained_state_dict, strict=False)

def build_finetune_model(finetune_args, 
                         cuda) -> nn.Module:
    device = 'cuda' if cuda else 'cpu'
    model_pretrained = torch.load(finetune_args.pretrained_model_path, map_location=device)
    state_dict_pretrained = model_pretrained['state_dict'] if finetune_args.load_pretrained else None
    grover_args = model_pretrained['args']
    setattr(grover_args, 'dropout', 0.1)
    setattr(grover_args, 'cuda', cuda)
    
    model = GROVERFinetune(finetune_args, grover_args)
    if cuda:
        model.cuda()
    initialize_model(model, state_dict_pretrained)
    return model

def get_optimizer(model, training_args):
    base_layers = model.grover.parameters()
    ffn_layers = chain(model.mol_atom_from_atom_ffn.parameters(), 
                       model.mol_atom_from_bond_ffn.parameters())
    optimizer = torch.optim.Adam([{'params': base_layers, 
                                   'lr': training_args.init_lr*training_args.finetune_coff},
                                 {'params': ffn_layers, 'lr': training_args.init_lr}],
                                 lr=training_args.init_lr,
                                 weight_decay=training_args.weight_decay)
    return optimizer

def get_dataset(data_args):
    with open(data_args.data_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        lines = [line for line in csv_reader]
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
        
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    return save_dir

def save_args(save_dir: str, **argument_sets):
    for key, arg_set in argument_sets.items():
        if not isinstance(arg_set, dict):
            argument_sets[key]=arg_set._asdict()
    with open(os.path.join(save_dir, 'arguments.txt'), 'w') as file:
        json.dump(argument_sets, file, indent=4)