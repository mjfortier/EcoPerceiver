import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import math
import yaml
import numpy as np
import random
from pathlib import Path

import timm.optim.optim_factory as optim_factory
from ecoperceiver.dataset import EcoPerceiverLoaderConfig, EcoPerceiverDataset
from ecoperceiver.components import EcoPerceiverConfig
from ecoperceiver.model import EcoPerceiver

import misc


def get_args_parser():
    parser = argparse.ArgumentParser('Perceiver Distributed Trainer', add_help=False)
    parser.add_argument('--run_dir', default='./runs/default')
    parser.add_argument('--config', default='config.yml')
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--tensorboard_dir', default='./tensorboard/default')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    return parser


def main(args):
    ############
    # Misc setup
    ############
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device('cuda')

    cudnn.benchmark = True

    # Make tensorboard directory
    if args.rank == 0:
        os.makedirs(args.tensorboard_dir, exist_ok=True)

    # Copy data to each node
    data_dir = args.data_dir
    tar_name = config['run']['tar_name']
    tar_file = os.path.join(data_dir, tar_name)
    slurm_path = os.path.join(os.environ["SLURM_TMPDIR"], 'data')
    slurm_tar_file = os.path.join(slurm_path, tar_name)
    if args.local_rank == 0:
        os.system(f"mkdir {slurm_path}")
        os.system(f'rsync -av --progress {tar_file} {slurm_tar_file}')
        os.system(f'tar xf {slurm_tar_file} -C {slurm_path}')
        os.system(f'rm -f {slurm_tar_file}')
    data_dir = os.path.join(slurm_path, 'carbonsense_v2')
    dist.barrier()


    ############
    # Dataloader
    ############
    individual_batch_size = config['dataloader']['batch_size']

    # Reproducibility 
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(SEED)

    dataset_config = EcoPerceiverLoaderConfig(**config['dataset'])
    dataset_train = EcoPerceiverDataset(data_dir, dataset_config, config['train_sites'])
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=args.world_size, rank=args.rank, shuffle=True
    )
    data_loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=individual_batch_size,
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory'],
        worker_init_fn=seed_worker,
        collate_fn=dataset_train.collate_fn,
        generator=g)

    dataset_val = EcoPerceiverDataset(data_dir, dataset_config, config['val_sites'])
    sampler_val = torch.utils.data.DistributedSampler(
        dataset_val, num_replicas=args.world_size, rank=args.rank
    )
    data_loader_val = DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=individual_batch_size,
        num_workers=config['dataloader']['num_workers'],
        pin_memory=config['dataloader']['pin_memory'],
        worker_init_fn=seed_worker,
        collate_fn=dataset_train.collate_fn,
        generator=g)

    tbdir = os.path.join(args.run_dir, 'tensorboard')
    if args.rank == 0:
        log_writer = SummaryWriter(log_dir=tbdir)
    else:
        log_writer = None
    

    #######
    # Model
    #######
    model_config = EcoPerceiverConfig(**config['model'])
    model = EcoPerceiver(model_config)
    model.to(device)
    
    #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module


    ###########
    # Optimizer
    ###########
    eff_batch_size = individual_batch_size * config['optimizer']['accum_iter'] * args.world_size
    lr = config['optimizer']['lr'] * eff_batch_size / 256
    config['optimizer']['lr'] = lr # cache this for later
    
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, config['optimizer']['weight_decay'])
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
    loss_scaler = misc.NativeScalerWithGradNormCount()
    misc.load_model(args=args, config=config, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    dist.barrier()

    # set global data type usage
    global datatype
    cuda_major = torch.cuda.get_device_properties(device).major
    if cuda_major >= 8: # supports bfloat16
        datatype = torch.bfloat16
    else:
        datatype = torch.float16
    
    ############
    # Train loop
    ############
    for epoch in range(config['optimizer']['start_epoch'], config['optimizer']['num_epochs']):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(
            model, data_loader_train, optimizer, epoch, loss_scaler, log_writer, config['optimizer']
        )
        validate_one_epoch(model, data_loader_val, epoch, log_writer)

        if args.rank == 0:
            if log_writer is not None:
                log_writer.flush()
                os.system(f"rsync -av {os.path.join(tbdir, '*')} {args.tensorboard_dir}")
            if epoch % config['optimizer']['save_interval'] == 0 or epoch + 1 == config['optimizer']['num_epochs']:
                misc.save_model(args, epoch, model_without_ddp, optimizer, loss_scaler)
        dist.barrier()


def adjust_learning_rate(optimizer, epoch, optim_config):
    """Decay the learning rate with half-cycle cosine after warmup"""
    min_lr = optim_config['min_lr']
    lr = optim_config['lr']
    warmup_epochs = optim_config['warmup_epochs']
    epochs = optim_config['num_epochs']

    if epoch < warmup_epochs:
        lr = lr * epoch / warmup_epochs 
    else:
        lr = min_lr + (lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def train_one_epoch(model, data_loader, optimizer, epoch, loss_scaler, log_writer, optim_config):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    accum_iter = optim_config['accum_iter']
    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq=1000, header=f'Epoch: [{epoch}]')):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, optim_config)
        
        with torch.cuda.amp.autocast(dtype=datatype):
            op, loss = model(batch)
        
        assert math.isfinite(loss.item()), f'Loss is {loss}, stopping training.'

        loss /= accum_iter

        # loss_scaler calls backward() and step()
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        
        torch.cuda.synchronize()
        metric_logger.update(loss=loss)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        loss_value_reduce = misc.all_reduce_mean(loss).item()
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
        
    metric_logger.synchronize_between_processes()
    if log_writer is not None:
        avg_loss = metric_logger.loss.global_avg
        log_writer.add_scalar('train_batch_loss', avg_loss, epoch)
        print(f'* loss {avg_loss:.3f}')
    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    

def validate_one_epoch(model, data_loader, epoch, log_writer):
    metric_logger = misc.MetricLogger(delimiter="  ")
    model.eval()
    
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq=1000, header='Val: ')):
        with torch.cuda.amp.autocast(dtype=datatype):
            op, loss = model(batch)

        assert math.isfinite(loss.item()), f'Loss is {loss}, stopping training.'
        metric_logger.update(loss=loss)

        loss_value_reduce = misc.all_reduce_mean(loss).item()
        if log_writer is not None:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('val_loss', loss_value_reduce, epoch_1000x)
    
    metric_logger.synchronize_between_processes()
    if log_writer is not None:
        avg_loss = metric_logger.loss.global_avg
        log_writer.add_scalar('val_batch_loss', avg_loss, epoch)
        print(f'* loss {avg_loss:.3f}')

    # avg = {f"{k}": meter.global_avg for k, meter in metric_logger.meters.items()}
    # std = {f"{k}-std": meter.std for k, meter in metric_logger.meters.items()}
    # avg.update(std)
    # return avg_loss
    

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
