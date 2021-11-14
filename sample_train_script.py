from GROVER_finetune.model import GROVERFinetune
from GROVER_finetune.utils import global_initiate, build_finetune_model, get_dataset
from GROVER_finetune.train import train
from GROVER_finetune.arguments import DataArgs, FinetuneArgs, TrainingArgs

data_args = DataArgs(
    data_path='exampledata/finetune/bbbp.csv',
    bond_drop_rate=0,
    no_cache=True)

finetune_args = FinetuneArgs(
    self_attn_hidden=4,
    self_attn_out=128,
    ffn_n_layer=2,
    ffn_hidden=128,
    ffn_out=1,
    ffn_activation='PReLU',
    classification=True, # if False then do regression.
    disagr_coff=1,
    pretrained_model_path='models_pretrained/grover_base.pt',
    load_pretrained=True) 

training_args = TrainingArgs(
    epochs=30,
    warmup_epochs=2,
    # train_data_size=len(valid_loader),
    init_lr=1e-4,
    max_lr=1e-3,
    final_lr=1e-4,
    weight_decay=0,
    finetune_coff=1,

    batch_size=32,
    eval_frequency=2,
    checkpoint_dir='train_checkpoint/load_pretrained',
    prefix_time_to_checkpoint_dir=True)

if __name__=='__main__':
    CUDA = global_initiate()
    model = build_finetune_model(finetune_args, cuda=CUDA)
    mol_dataset = get_dataset(data_args)
    train(model, mol_dataset, training_args, data_args, cuda=CUDA)