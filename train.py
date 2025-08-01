# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import git
import math
import argparse

import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision.utils as vutils

from model import PSBAutoEncoder

from data import GlobVideoDataset

from utils import linear_warmup, all_reduce

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--num_workers', type=int, default=24)
parser.add_argument('--epochs', type=int, default=500)

parser.add_argument('--world_size', type=int, default=1)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--addr', default='localhost')
parser.add_argument('--port', default='12900')
parser.add_argument('--backend', default='nccl')

parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--image_channels', type=int, default=3)
parser.add_argument('--ep_len', type=int, default=6)

parser.add_argument('--checkpoint_path', default='checkpoint.pt.tar')
parser.add_argument('--data_path', default='training_data/*')
parser.add_argument('--log_path', default='logs/')

parser.add_argument('--d_model', type=int, default=192)
parser.add_argument('--d_scalar', type=int, default=16)

parser.add_argument('--num_slots', type=int, default=11)
parser.add_argument('--psb_num_blocks', type=int, default=3)
parser.add_argument('--psb_num_heads', type=int, default=4)

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_warmup_steps', type=int, default=30000)
parser.add_argument('--lr_half_life', type=int, default=1000000)
parser.add_argument('--clip', type=float, default=0.05)

parser.add_argument('--use_ddp', default=True, action='store_true')

args = parser.parse_args()

torch.manual_seed(args.seed)

if args.rank == 0:
    arg_str_list = ['{}={}'.format(k, v) for k, v in vars(args).items()]
    arg_str = '__'.join(arg_str_list)
    log_dir = args.log_path
    writer = SummaryWriter(log_dir)
    writer.add_text('hparams', arg_str)

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    writer.add_text('git', sha)

if args.use_ddp:
    from torch.utils.data.distributed import DistributedSampler
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.distributed import init_process_group, destroy_process_group, reduce

    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = args.port
    init_process_group(backend=args.backend, rank=args.rank, world_size=args.world_size)


def visualize(video, recon, masks):
    B, T, K, C, H, W = video.size()

    frames = []
    for t in range(T):
        video_t = video[:B, t, :, :, :, :]
        recon_t = recon[:B, t, :, :, :, :]
        masks_t = masks[:B, t, :, :, :, :]

        # tile
        tiles = torch.cat((video_t, recon_t, masks_t), dim=1).flatten(end_dim=1)

        # grid
        frame = vutils.make_grid(tiles, nrow=tiles.shape[0] // B, pad_value=0.8)
        frames += [frame]

    frames = torch.stack(frames, dim=0).unsqueeze(0)

    return frames


train_dataset = GlobVideoDataset(root=args.data_path, phase='train', image_size=args.image_size, ep_len=args.ep_len, image_glob='????????_image.png')
val_dataset = GlobVideoDataset(root=args.data_path, phase='val', image_size=args.image_size, ep_len=args.ep_len, image_glob='????????_image.png')

if args.use_ddp:
    train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=args.seed, drop_last=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=True, seed=args.seed, drop_last=True)
else:
    train_sampler = None
    val_sampler = None

loader_kwargs = {
    'batch_size': args.batch_size,
    'shuffle': not args.use_ddp,
    'num_workers': args.num_workers,
    'pin_memory': True,
    'drop_last': True,
}

train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)
val_loader = DataLoader(val_dataset, sampler=val_sampler, **loader_kwargs)

train_epoch_size = len(train_loader)
val_epoch_size = len(val_loader)

log_interval = train_epoch_size // 5

model = PSBAutoEncoder(args)

if os.path.isfile(args.checkpoint_path):
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    best_epoch = checkpoint['best_epoch']
    model.load_state_dict(checkpoint['model'])
else:
    checkpoint = None
    start_epoch = 0
    best_val_loss = math.inf
    best_epoch = 0

model = model.cuda()
if args.use_ddp:
    model = DDP(model)

lr = args.lr
optimizer = AdamW([
    {'params': (x[1] for x in model.named_parameters()), 'lr': lr, 'betas':(0.9, 0.95)},
])
scaler = torch.cuda.amp.GradScaler()
if checkpoint is not None:
    optimizer.load_state_dict(checkpoint['optimizer'])
    scaler.load_state_dict(checkpoint['scaler'])

for epoch in range(start_epoch, args.epochs):
    model.train()

    for batch, video in enumerate(train_loader):
        global_step = epoch * train_epoch_size + batch

        lr_warmup_factor = linear_warmup(
            global_step,
            0.,
            1.0,
            0.,
            args.lr_warmup_steps)

        lr_decay_factor = math.exp(global_step / args.lr_half_life * math.log(0.5))

        optimizer.param_groups[0]['lr'] = lr_decay_factor * lr_warmup_factor * lr

        video = video.cuda()

        with torch.cuda.amp.autocast(True):
            loss, slots, recons, probs, recon = model(video)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_reduce = all_reduce(loss) / args.world_size if args.use_ddp else loss

        with torch.no_grad():
            if args.rank == 0:
                if batch % log_interval == 0:
                    print('Train Epoch: {:3} [{:5}/{:5}] \t Loss: {:F}'.format(
                        epoch + 1, batch, train_epoch_size, loss_reduce.item()))

                    writer.add_scalar('TRAIN/loss', loss_reduce.item(), global_step)
                    writer.add_scalar('TRAIN/lr', optimizer.param_groups[0]['lr'], global_step)

    with torch.no_grad():
        if args.rank == 0:
            B = 8
            frames = visualize(video[:B, :, None, :, :, :], recon[:B, :, None, :, :, :], probs[:B, :, :, :, :, :].repeat(1, 1, 1, 3, 1, 1))
            writer.add_video('TRAIN_recons/epoch={:03}'.format(epoch + 1), frames, fps=1)

    with torch.no_grad():
        model.eval()

        val_loss = 0.

        for batch, video in enumerate(val_loader):
            video = video.cuda()

            with torch.cuda.amp.autocast(True):
                loss, *_ = model(video)

            loss = all_reduce(loss) / args.world_size if args.use_ddp else loss

            val_loss += loss.item()

        val_loss /= (val_epoch_size)

        if args.rank == 0:

            writer.add_scalar('VAL/loss', val_loss, epoch + 1)

            print('====> Epoch: {:3} \t Loss = {:F}'.format(epoch + 1, val_loss))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1

                torch.save(model.module.state_dict() if args.use_ddp else model.state_dict(),
                           os.path.join(log_dir, 'best_model.pt'))

                if 50 <= epoch:
                    pass

            writer.add_scalar('VAL/best_loss', best_val_loss, epoch + 1)

            checkpoint = {
                'epoch': epoch + 1,
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'model': model.module.state_dict() if args.use_ddp else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
            }

            torch.save(checkpoint, os.path.join(log_dir, 'checkpoint.pt.tar'))

            print('====> Best Loss = {:F} @ Epoch {}'.format(best_val_loss, best_epoch))

if args.rank == 0:
    writer.close()

if args.use_ddp:
    destroy_process_group()
