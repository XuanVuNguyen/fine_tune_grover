from GROVER_finetune.utils import global_initiate, unpack_checkpoint_dir
from GROVER_finetune.train_and_test import test
from GROVER_finetune.arguments import DataArgs, EvaluationArgs

evaluation_args = EvaluationArgs(
    checkpoint_dir='train_checkpoint/2021_11_14-16_38_53-no_load_pretrained',
    epoch=28)

if __name__=='__main__':
    CUDA = global_initiate()
    model, data_args = unpack_checkpoint_dir(evaluation_args, cuda=CUDA)
    test(model, evaluation_args, data_args, cuda=CUDA)