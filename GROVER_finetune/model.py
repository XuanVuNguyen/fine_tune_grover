from collections import OrderedDict
import torch
from torch import nn
from torch.utils.data import DataLoader

from grover.model.models import GROVEREmbedding, Readout
from grover.util.nn_utils import get_activation_function

class GROVERFinetune(nn.Module):
    def __init__(self, finetune_args, grover_args):
        super().__init__()
        self.finetune_args = finetune_args
        self.grover_args = grover_args
        self.grover = GROVEREmbedding(grover_args)
        self.readout = Readout(rtype="mean",
                               hidden_size=grover_args.hidden_size)
            # rtype='self_attention',
            # hidden_size=grover_args.hidden_size,
            # attn_hidden=finetune_args.self_attn_hidden,
            # attn_out=finetune_args.self_attn_out) # readout is shared between two views of atom embedded.
    
        self.mol_atom_from_atom_ffn = self._create_ffn(finetune_args, grover_args)
        self.mol_atom_from_bond_ffn = self._create_ffn(finetune_args, grover_args)
        self.sigmoid = nn.Sigmoid()
        
    def _create_ffn(self, finetune_args, grover_args):
        
        input_dim = grover_args.hidden_size + finetune_args.append_mol_features*200 #dim of mol features by rdkit_2d_normalized. 
            # * finetune_args.self_attn_out
        # dropout = nn.Dropout(p=grover_args.dropout)
        # activation = get_activation_function(finetune_args.ffn_activation)
        
#         ffn = []
#         if finetune_args.ffn_n_layer == 1:
#             ffn.extend(
#                 [nn.Dropout(p=grover_args.dropout),
#                  nn.Linear(input_dim, finetune_args.ffn_out)
#                 ]
#             )
#         else:
#             ffn.extend(
#                 [nn.Dropout(p=grover_args.dropout), # shared dropout?
#                  nn.Linear(input_dim, finetune_args.ffn_hidden)
#                 ]
#             )

#             for i in range(finetune_args.ffn_n_layer-2):
#                 ffn.extend(
#                     [get_activation_function(finetune_args.ffn_activation),
#                      nn.Dropout(p=grover_args.dropout),
#                      nn.Linear(finetune_args.ffn_hidden, finetune_args.ffn_hidden)
#                     ]
#                 )
            
#             ffn.extend(
#                 [get_activation_function(finetune_args.ffn_activation),
#                  nn.Dropout(p=grover_args.dropout),
#                  nn.Linear(finetune_args.ffn_hidden, finetune_args.ffn_out)
#                 ]
#             )
            
        ffn = OrderedDict()
        if finetune_args.ffn_n_layer == 1:
            ffn.update(
                {'dropout':nn.Dropout(p=grover_args.dropout),
                 'fc':nn.Linear(input_dim, finetune_args.ffn_out)
                }
            )
        else:
            ffn.update(
                {'dropout_0':nn.Dropout(p=grover_args.dropout), # shared dropout?
                 'fc_0':nn.Linear(input_dim, finetune_args.ffn_hidden)
                }
            )
            next_layer_id = 1
            for i in range(finetune_args.ffn_n_layer-2):
                ffn.update(
                    {f'act_func_{i}':get_activation_function(finetune_args.ffn_activation),
                     f'dropout_{i+1}':nn.Dropout(p=grover_args.dropout),
                     f'fc_{i+1}':nn.Linear(finetune_args.ffn_hidden, finetune_args.ffn_hidden)
                    }
                )
                next_layer_id += 1
            
            ffn.update(
                {f'act_func_{next_layer_id-1}':get_activation_function(finetune_args.ffn_activation),
                 f'dropout_{next_layer_id}':nn.Dropout(p=grover_args.dropout),
                 f'fc_{next_layer_id}':nn.Linear(finetune_args.ffn_hidden, finetune_args.ffn_out)
                }
            )

        return nn.Sequential(ffn)
    
    def forward(self, batch, mol_feature_batch):
        a_scope = batch[5]
        grover_output = self.grover(batch)
        mol_atom_from_atom = self.readout(grover_output['atom_from_atom'], a_scope)
        mol_atom_from_bond = self.readout(grover_output['atom_from_bond'], a_scope)
        if self.finetune_args.append_mol_features:
            assert mol_feature_batch is not None, "append_mol_features_batch is True but mol_feature_batch is empty."
            mol_atom_from_atom = torch.cat([mol_atom_from_atom, mol_feature_batch], -1)
            mol_atom_from_bond = torch.cat([mol_atom_from_bond, mol_feature_batch], -1)
        atom_ffn_output = self.mol_atom_from_atom_ffn(mol_atom_from_atom)
        bond_ffn_output = self.mol_atom_from_bond_ffn(mol_atom_from_bond)
        if self.training:
            return atom_ffn_output, bond_ffn_output, mol_atom_from_atom
        else:
            if self.finetune_args.classification:
                atom_ffn_output = self.sigmoid(atom_ffn_output)
                bond_ffn_output = self.sigmoid(bond_ffn_output)
            output = (atom_ffn_output + bond_ffn_output) / 2
            return output
        
    @staticmethod
    def get_loss_fn(finetune_args):
        def finetune_loss_fn(preds, targets,
                            classification=finetune_args.classification,
                            disagr_coff=finetune_args.disagr_coff):
            if classification:
                pred_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
            else:
                pred_loss_fn = nn.MSELoss(reduction='none')
            
            if type(preds) is not tuple: # eval mode
                return pred_loss_fn(preds, targets)
            
            # training mode
            disagr_loss_fn = nn.MSELoss(reduction='none')
            pred_loss_1 = pred_loss_fn(preds[0], targets)
            pred_loss_2 = pred_loss_fn(preds[1], targets)
            disagr_loss = disagr_loss_fn(preds[0], preds[1])
            
            final_loss = pred_loss_1 + pred_loss_2 + disagr_coff*disagr_loss
            return final_loss
        
        return finetune_loss_fn