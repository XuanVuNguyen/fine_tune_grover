from collections import namedtuple

DataArgs = namedtuple('DataArgs',
                       ['data_path',
                        'bond_drop_rate',
                        'no_cache'])

FinetuneArgs = namedtuple('FinetuneArgs',
                          ['self_attn_hidden',
                           'self_attn_out',
                           'ffn_n_layer',
                           'ffn_hidden',
                           'ffn_out',
                           'ffn_activation',
                           'classification',
                           'disagr_coff',
                           'pretrained_model_path',
                           'load_pretrained'])

TrainingArgs = namedtuple('TrainingArgs', 
                          ['epochs',
                           'warmup_epochs',
                           # 'train_data_size',
                           'init_lr',
                           'max_lr',
                           'final_lr',
                           'weight_decay',
                           'finetune_coff',
                           'batch_size',
                           'eval_frequency',
                           'checkpoint_dir',
                           'prefix_time_to_checkpoint_dir'])