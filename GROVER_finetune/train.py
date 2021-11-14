import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from rdkit import RDLogger

import time
from tqdm import tqdm
import os

from GROVER_finetune.utils import get_optimizer, create_save_dir, save_args

from grover.util.utils import scaffold_split
from grover.util.scheduler import NoamLR
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
        loss = loss.sum() / mask.sum()
        epoch_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
        if isinstance(lr_scheduler, NoamLR):
            lr_scheduler.step()
 
    return epoch_loss/len(train_loader)

def evaluate(model, valid_data_loader, loss_fn, cuda):
    model.eval()
    total_loss = 0
    if model.finetune_args.classification:
        correct_pred_count = 0
    with torch.no_grad():
        for step, batch_item in enumerate(valid_data_loader):
            _, batch, features_batch, mask, target = batch_item
            if cuda:
                mask, target = mask.cuda(), target.cuda()
                # batch is send to cuda by the model internally.
            pred = model(batch)
            loss = loss_fn(pred, target)
            loss = loss.sum()/mask.sum()
            if model.finetune_args.classification:
                batch_correct = ((pred>0.5).type(torch.float) == target).type(torch.float).sum().item()
                correct_pred_count += batch_correct
            total_loss += loss.item()
    eval_loss = total_loss/len(valid_data_loader)
    if model.finetune_args.classification:
        accuracy = correct_pred_count/len(valid_data_loader.dataset)
        return eval_loss, accuracy
    else:
        return eval_loss

def train(model, dataset,
          training_args, data_args, cuda):
    checkpoint_save_dir = create_save_dir(training_args.checkpoint_dir, training_args.prefix_time_to_checkpoint_dir)
    save_args(checkpoint_save_dir, 
              finetune_args=model.finetune_args,
              training_args=training_args,
              data_args=data_args
             )
    writer = SummaryWriter(log_dir=checkpoint_save_dir)
    
    # split data based on molecular scaffold
    train_dataset, valid_dataset, _ = scaffold_split(dataset)
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
        batch_size=training_args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=mol_collator)
    
    loss_fn = model.get_loss_fn(model.finetune_args)
    optimizer = get_optimizer(model, training_args)
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
        writer.add_scalar('lr/base', optimizer.param_groups[0]['lr'], epoch_count)
        writer.add_scalar('lr/ffn', optimizer.param_groups[1]['lr'], epoch_count)
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
                cuda=cuda) # done
            if type(eval_performance) is tuple:
                writer.add_scalar('Evaluation/loss', eval_performance[0], epoch)
                writer.add_scalar('Evaluation/accuracy', eval_performance[1], epoch)
                eval_loss = eval_performance[0]
                print('======= eval_loss: {:.4f}, eval_accuracy: {:.4f}'.format(*eval_performance))
            else:    
                writer.add_scalar('Evaluation loss', eval_performance, epoch)
                eval_loss = eval_performance
                print('======= eval_loss: {:.4f}'.format(eval_performance))
            
            for name, weight in model.named_parameters():
                writer.add_histogram(os.path.join('weight_distribution', name), weight, epoch_count)
            
            checkpoint_file = 'epoch-{:03}-eval_loss-{:.3f}.pt'.format(epoch_count, eval_loss)
            save_path = os.path.join(checkpoint_save_dir, checkpoint_file)
            torch.save(model.state_dict(), save_path)
            print('======= Checkpoint saved at: ' + training_args.checkpoint_dir)
    # writer.flush()
    writer.close()