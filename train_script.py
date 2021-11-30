from GROVER_finetune.utils import global_initiate, build_finetune_model
from GROVER_finetune.train_and_test import train
from GROVER_finetune.arguments import DataArgs, FinetuneArgs, TrainingArgs

data_args = DataArgs(
    data_path='exampledata/finetune/bace.csv',
    scaffold_split_balanced=True, # switch to True if run classification.
    bond_drop_rate=0,
    no_cache=True)

finetune_args = FinetuneArgs(
    # self_attn_hidden=128, # this argument is not relevant anymore
    # self_attn_out=4, # this argument is not relevant anymore
    ffn_n_layer=3,
    ffn_hidden=200,
    ffn_out=1,
    ffn_activation='ReLU',
    classification=True, # if False then do regression.
    disagr_coff=0.15,
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
    # finetune_coff=1, # this argument is not relevant anymore

    batch_size=32,
    eval_frequency=1,
    metric='auc',
    checkpoint_dir='train_checkpoint/bace_no_load_pretrained_bias_init_0.1_relu',
    prefix_time_to_checkpoint_dir=True)

if __name__=='__main__':
    CUDA = global_initiate()
    model = build_finetune_model(finetune_args, cuda=CUDA)
    train(model, training_args, data_args, cuda=CUDA)