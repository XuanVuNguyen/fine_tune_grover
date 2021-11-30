import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from rdkit import RDLogger

import time
from tqdm import tqdm
import os

from GROVER_finetune.utils import get_optimizer, create_save_dir, save_args, get_dataset

from grover.util.utils import scaffold_split
from grover.util.scheduler import NoamLR
from grover.util.metrics import get_metric_func
from grover.data.molgraph import MolCollator

# disable rdkit warning
RDLogger.DisableLog('rdApp.*')

def train_epoch(model, epoch, total_epoch, train_loader,
               loss_fn, optimizer, lr_scheduler, cuda):

    model.train()
    epoch_loss = 0
    for step, batch_item in enumerate(tqdm(train_loader, desc='Epoch {}/{}'.format(epoch, total_epoch))):
        _, batch, features_batch, mask, target = batch_item
        if cuda:
            mask, target = mask.cuda(), target.cuda()
        model.zero_grad()
        
        pred = model(batch)
        loss = loss_fn(pred, target)
        loss = loss.sum()
        epoch_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
        if isinstance(lr_scheduler, NoamLR):
            lr_scheduler.step()
 
    return epoch_loss/len(train_loader)

def evaluate(model, valid_data_loader, loss_fn, metric_fn, cuda):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    if model.finetune_args.classification:
        correct_pred_count = 0
    with torch.no_grad():
        for step, batch_item in enumerate(valid_data_loader):
            _, batch, features_batch, mask, target = batch_item
            if cuda:
                mask, target = mask.cuda(), target.cuda()
                # batch is send to cuda by the model internally.
            pred = model(batch)
            
            # if model.finetune_args.classification: # accumulate pred and target to calculate roc auc score
            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())
            loss = loss_fn(pred, target)
            loss = loss.sum()
            if model.finetune_args.classification:
                batch_correct = ((pred>0.5).type(torch.float) == target).type(torch.float).sum().item()
                correct_pred_count += batch_correct
            total_loss += loss.item()
    eval_loss = total_loss/len(valid_data_loader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    metric = metric_fn(all_targets, all_preds)
    if model.finetune_args.classification:
        accuracy = correct_pred_count/len(valid_data_loader.dataset)
        return {'loss': eval_loss, 'accuracy': accuracy, metric_fn.__name__: metric}
    else:
        return {'loss': eval_loss, metric_fn.__name__: metric}

def train(model,
          training_args, data_args, cuda):
    checkpoint_save_dir = create_save_dir(training_args.checkpoint_dir, training_args.prefix_time_to_checkpoint_dir)
    save_args(checkpoint_save_dir, 
              finetune_args=model.finetune_args,
              training_args=training_args,
              data_args=data_args
             )
    writer = SummaryWriter(log_dir=checkpoint_save_dir)
    dataset = get_dataset(data_args)
    # split data based on molecular scaffold
    train_dataset, valid_dataset, _ = scaffold_split(dataset, balanced=data_args.scaffold_split_balanced)
    # create data loader
    mol_collator = MolCollator({}, data_args)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=training_args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=mol_collator)
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=mol_collator)
    
    loss_fn = model.get_loss_fn(model.finetune_args)
    optimizer = get_optimizer(model, training_args)
    metric_fn = get_metric_func(training_args.metric)
    lr_scheduler = NoamLR(optimizer=optimizer,
                          warmup_epochs=training_args.warmup_epochs,
                          total_epochs=training_args.epochs,
                          steps_per_epoch=len(train_loader),
                          init_lr=training_args.init_lr,
                          max_lr=training_args.max_lr,
                          final_lr=training_args.final_lr)
    
    global_time_start = time.time()
    
    # write model graph to tensorboard
    # train_iter = iter(train_loader)
    # _, random_batch, _, _, _ = train_iter.__next__()
    # writer.add_graph(model, (random_batch, ), verbose=True)
    
    for epoch in range(training_args.epochs):
        epoch_count = epoch + 1
        epoch_start = time.time()
        for i, param_group in enumerate(optimizer.param_groups):
            writer.add_scalar(f'lr/{i}', param_group['lr'], epoch_count)
            # writer.add_scalar('lr/ffn', optimizer.param_groups[1]['lr'], epoch_count)
        epoch_average_loss = train_epoch(model, epoch_count, training_args.epochs,
                                         train_loader,
                                         loss_fn=loss_fn,
                                         optimizer=optimizer,
                                         lr_scheduler=lr_scheduler,
                                         cuda=cuda) # done
        epoch_run_time = time.time() - epoch_start
        writer.add_scalar('Train loss', epoch_average_loss, epoch_count, time.time()-global_time_start)
        
        print('Epoch {}/{}: train_loss: {:.4f}, time: {:.2f}'.format(epoch_count, 
                                                                     training_args.epochs,
                                                                     epoch_average_loss,
                                                                     epoch_run_time))
         
        if (epoch_count) % training_args.eval_frequency==0:
            eval_performance = evaluate(
                model, valid_loader,
                loss_fn=loss_fn,
                metric_fn=metric_fn,
                cuda=cuda) # done
            # if type(eval_performance) is tuple:
            #     writer.add_scalar('Evaluation/loss', eval_performance[0], epoch_count)
            #     writer.add_scalar('Evaluation/accuracy', eval_performance[1], epoch_count)
            #     eval_loss = eval_performance[0]
            #     print('======= eval_loss: {:.4f}, eval_accuracy: {:.4f}'.format(*eval_performance))
            # else:    
            #     writer.add_scalar('Evaluation loss', eval_performance, epoch_count)
            #     eval_loss = eval_performance
            #     print('======= eval_loss: {:.4f}'.format(eval_performance))
            print('======= Evaluation:')
            for metric, value in eval_performance.items():
                writer.add_scalar(f'Evaluation/{metric}', value, epoch_count)
                print(f'======= {metric}: ', value)
                
            for name, weight in model.named_parameters():
                if weight.numel() > 0:
                    writer.add_histogram(os.path.join('weight_distribution', name), weight, epoch_count)
            
            checkpoint_file = 'epoch-{:03}-eval_loss-{:.3f}.pt'.format(epoch_count, eval_performance['loss'])
            save_path = os.path.join(checkpoint_save_dir, 'weight', checkpoint_file)
            torch.save(model.state_dict(), save_path)
            print(f'======= Checkpoint saved at: {training_args.checkpoint_dir}/weight')
    # writer.flush()
    writer.close()
    
def test(model, evaluation_args, data_args, cuda):
    mol_dataset = get_dataset(data_args)
    _, _, test_dataset = scaffold_split(mol_dataset, balanced=data_args.scaffold_split_balanced)
    mol_collator = MolCollator({}, data_args)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        collate_fn=mol_collator)
    loss_fn = model.get_loss_fn(model.finetune_args)
    metric_fn = get_metric_func(evaluation_args.metric)
    performance = evaluate(model, test_loader, loss_fn, metric_fn, cuda)
#     if type(performance) is tuple:
#         print('Loss: ', performance[0])
#         print('Accuracy: ', performance[1])
#         with open(os.path.join(evaluation_args.checkpoint_dir, 'test_set_performance.txt'), 'w') as file:
#             file.write('Loss: {}\n'.format(performance[0]))
#             file.write('Accuracy: {}'.format(performance[1]))
    
#     else:
#         print('Loss: ', performance)
#         with open(os.path.join(evaluation_args.checkpoint_dir, 'test_set_performance.txt'), 'w') as file:
#             file.write('Loss: {}\n'.format(performance))
    with open(os.path.join(evaluation_args.checkpoint_dir, 'test_set_performance.txt'), 'w') as file:
        file.write(f'Epoch {evaluation_args.epoch}\n')
    print(f'Epoch {evaluation_args.epoch}')
    for metric, value in performance.items():
        with open(os.path.join(evaluation_args.checkpoint_dir, 'test_set_performance.txt'), 'a') as file:
            file.write(f'{metric}: {value}\n')
        print(f'{metric}: {value}')