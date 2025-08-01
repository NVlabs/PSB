# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from utils import *
from psb import PSB


class CartesianPositionalEmbedding(nn.Module):
    def __init__(self, channels, image_size):
        super().__init__()

        self.projection = conv2d(4, channels, 1)
        self.pe = nn.Parameter(
            self.build_grid(image_size).unsqueeze(0), requires_grad=False
        )

    def build_grid(self, side_length):
        coords = torch.linspace(-math.sqrt(2), math.sqrt(2), side_length + 1)
        coords = 0.5 * (coords[:-1] + coords[1:])
        grid_y, grid_x = torch.meshgrid(coords, coords)
        return torch.stack((grid_x, grid_y, -grid_x, -grid_y), dim=0)

    def forward(self, inputs):
        # `inputs` has shape: [batch_size, out_channels, height, width].
        # `grid` has shape: [batch_size, in_channels, height, width].
        return inputs + self.projection(self.pe)


class SpatialBroadcastDecoder(nn.Module):
    def __init__(self, image_size, slot_size, output_size, broadcast_size=8):
        super().__init__()
        self.broadcast_size = broadcast_size

        self.slots_norm = nn.LayerNorm(slot_size)

        self.decoder_pos = CartesianPositionalEmbedding(slot_size, broadcast_size)
        self.decoder_cnn = nn.Sequential(
            nn.GroupNorm(1, slot_size),
            nn.ConvTranspose2d(slot_size, 64, 5, 2, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 5, 2, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 5, 2, 2, 1),
            nn.ReLU(),
            (
                nn.ConvTranspose2d(64, output_size, 5, 2, 2, 1)
                if image_size == 128
                else nn.ConvTranspose2d(64, output_size, 5, 1, 2, 0)
            ),
        )

    def forward(self, slots):
        """

        :param slots: B, D
        :return:
        """
        B, D = slots.size()

        slots = self.slots_norm(slots)  # B, D

        x = slots[:, :, None, None].expand(
            B, D, self.broadcast_size, self.broadcast_size
        )  # B, D, H, W
        x = self.decoder_pos(x)  # B, D, H, W
        x = self.decoder_cnn(x)  # B, out, H, W

        return x


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.d_model = args.d_model
        self.num_slots = args.num_slots
        self.image_channels = args.image_channels

        self.image_norm = NormalizePixel()

        self.slot_init = nn.Parameter(torch.Tensor(1, 1, self.num_slots, self.d_model))
        nn.init.trunc_normal_(self.slot_init)

        self.encoder_cnn = nn.Sequential(
            conv2d(args.image_channels, args.d_model, 5, 1, 2, weight_init="kaiming"),
            nn.ReLU(),
            conv2d(args.d_model, args.d_model, 5, 1, 2, weight_init="kaiming"),
            nn.ReLU(),
            conv2d(args.d_model, args.d_model, 5, 1, 2, weight_init="kaiming"),
            nn.ReLU(),
            conv2d(args.d_model, args.d_model, 5, 1, 2),
        )
        self.encoder_pos = CartesianPositionalEmbedding(args.d_model, args.image_size)

        self.layer_norm = nn.LayerNorm(args.d_model)
        self.mlp = nn.Sequential(
            linear(args.d_model, args.d_model, weight_init="kaiming"),
            nn.ReLU(),
            linear(args.d_model, args.d_model),
        )

        self.binder = PSB(
            args.d_model, args.psb_num_blocks, args.psb_num_heads
        )

    def forward(self, video):
        """

        :param video: B, T, C, H, W
        :return: slots: B, T, N, D
        """

        B, T, C, H, W = video.shape

        video_normed = self.image_norm(video.permute(0, 1, 3, 4, 2)).permute(
            0, 1, 4, 2, 3
        )  # B, T, C, H, W
        video_normed = video_normed.reshape(B * T, C, H, W)  # BT, C, H, W

        emb = self.encoder_cnn(video_normed)  # BT, D, H_enc, W_enc
        emb = self.encoder_pos(emb)  # BT, D, H_enc, W_enc

        emb = emb.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)  # BT, L, D
        emb = self.mlp(self.layer_norm(emb))  # BT, L, D

        BT, L, D = emb.shape

        emb = emb.reshape(B, T, L, D)  # B, T, L, D

        slots = self.slot_init.repeat(B, T, 1, 1)  # B, T, N, D

        slots = self.binder(slots, emb)  # B, T, N, D

        return slots


class PSBAutoEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.encoder = Encoder(args)
        self.decoder = SpatialBroadcastDecoder(
            args.image_size, args.d_model, args.image_channels + 1
        )

    def forward(self, video):
        """

        :param video: B, T, C, H, W
        :return:
        """
        B, T, C, H, W = video.shape

        slots = self.encoder(video)  # B, T, N, D
        _, _, N, D = slots.shape

        pred = self.decoder(slots.reshape(B * T * N, D)).reshape(
            B, T, N, C + 1, H, W
        )  # B, T, N, C + 1, H, W

        recons, logprobs = pred.split(
            [C, 1], dim=-3
        )  # B, T, N, C, H, W ; B, T, N, 1, H, W
        probs = torch.softmax(logprobs, dim=-4)  # B, T, N, 1, H, W

        recon = (probs * recons).sum(-4)  # B, T, C, H, W

        loss = (recon - video).pow(2).mean()  # 1

        return (
            loss,
            slots,
            recons,
            probs,
            recon,
        )
