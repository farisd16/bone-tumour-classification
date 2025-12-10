import os
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from latent_diffusion.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_EPOCH,
    DIFFUSION_STEP,
    BETA_START,
    BETA_END,
    IMAGE_SIZE,
    SCALE_DOWN,
    NUM_CLASSES,
)
from latent_diffusion.utils import AvgMeter, EMA


class DDPMScheduler(nn.Module):
    def __init__(
        self,
        T=DIFFUSION_STEP,
        beta_start=BETA_START,
        beta_end=BETA_END,
    ):
        super().__init__()

        self.T = T
        self.beta = torch.linspace(beta_start, beta_end, T).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def forward(self, x, t, noise=None):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = (
            torch.sqrt(1 - self.alpha_hat[t]).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        )

        if noise is None:
            noise = torch.randn_like(x)
            torch.nn.init.uniform_(noise, -1.0, 1.0)

        x_noisy = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise

        return x_noisy, noise


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super().__init__()
        self.channels = channels
        self.size = size

        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(
            -1,
            self.channels,
            self.size,
            self.size,
        )


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        residual=False,
    ):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(
            1,
            1,
            x.shape[-2],
            x.shape[-1],
        )
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(
            1,
            1,
            x.shape[-2],
            x.shape[-1],
        )
        return x + emb


class PositionalEmbedding(nn.Module):
    def __init__(
        self,
        channels,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.channels = channels
        self.device = device

    def forward(self, time):
        inv_freq = 1.0 / (
            10000
            ** (
                torch.arange(0, self.channels, 2, device=self.device).float()
                / self.channels
            )
        )
        pos_enc_a = torch.sin(time.repeat(1, self.channels // 2) * inv_freq)
        pos_enc_b = torch.cos(time.repeat(1, self.channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc


class UNet(nn.Module):
    def __init__(
        self,
        c_in=3,
        c_out=3,
        num_classes=NUM_CLASSES,
        time_dim=1280 // SCALE_DOWN,
        scale_factor=8,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()

        ##### Encoder
        self.input_conv = DoubleConv(c_in, 320 // SCALE_DOWN)

        self.down1 = Down(320 // SCALE_DOWN, 640 // SCALE_DOWN, time_dim)
        self.sa1 = SelfAttention(640 // SCALE_DOWN, (IMAGE_SIZE // scale_factor) // 2)

        self.down2 = Down(640 // SCALE_DOWN, 1280 // SCALE_DOWN, time_dim)
        self.sa2 = SelfAttention(
            1280 // SCALE_DOWN,
            (IMAGE_SIZE // scale_factor) // 4,
        )

        self.down3 = Down(1280 // SCALE_DOWN, 1280 // SCALE_DOWN, time_dim)
        self.sa3 = SelfAttention(
            1280 // SCALE_DOWN,
            (IMAGE_SIZE // scale_factor) // 8,
        )

        ##### Bridge or Bottleneck
        self.bridge = nn.Sequential(
            DoubleConv(1280 // SCALE_DOWN, 2560 // SCALE_DOWN),
            DoubleConv(2560 // SCALE_DOWN, 2560 // SCALE_DOWN),
            SelfAttention(2560 // SCALE_DOWN, (IMAGE_SIZE // scale_factor) // 8),
            DoubleConv(2560 // SCALE_DOWN, 1280 // SCALE_DOWN),
        )

        ##### Decoder
        self.up1 = Up(
            2560 // SCALE_DOWN,
            640 // SCALE_DOWN,
            time_dim,
        )
        self.sa4 = SelfAttention(640 // SCALE_DOWN, (IMAGE_SIZE // scale_factor) // 4)

        self.up2 = Up(
            1280 // SCALE_DOWN,
            320 // SCALE_DOWN,
            time_dim,
        )
        self.sa5 = SelfAttention(320 // SCALE_DOWN, (IMAGE_SIZE // scale_factor) // 2)

        self.up3 = Up(
            640 // SCALE_DOWN,
            320 // SCALE_DOWN,
            time_dim,
        )
        self.sa6 = SelfAttention(320 // SCALE_DOWN, (IMAGE_SIZE // scale_factor))

        self.out_conv = nn.Conv2d(320 // SCALE_DOWN, c_out, kernel_size=1)

        ##### Positional Embedding
        self.pos_encoding = PositionalEmbedding(time_dim, device)

        ##### Label Embedding
        self.label_emb = nn.Embedding(num_classes, time_dim)

    def forward(self, x, t, context=None):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t)

        if context is not None:
            t += self.label_emb(context)

        #### Encoder
        x = self.input_conv(x)

        x1 = self.down1(x, t)
        x1 = self.sa1(x1)

        x2 = self.down2(x1, t)
        x2 = self.sa2(x2)

        x3 = self.down3(x2, t)
        x3 = self.sa3(x3)

        #### Bridge or Bottleneck
        bridge = self.bridge(x3)

        #### Decoder
        out = self.up1(bridge, x2, t)
        out = self.sa4(out)

        out = self.up2(out, x1, t)
        out = self.sa5(out)

        out = self.up3(out, x, t)
        out = self.sa6(out)

        out = self.out_conv(out)

        return out


class LatentDiffusionWrapper(L.LightningModule):
    def __init__(self, vae_model, run_dir=None):
        super().__init__()

        self.run_dir = run_dir

        self.forward_model = DDPMScheduler()
        self.backward_model = UNet()

        self.vae_model = vae_model
        self.vae_model.eval()

        for param in self.vae_model.parameters():
            param.requires_grad = False

        self.ema = EMA()
        self.ema_model = copy.deepcopy(self.backward_model)
        self.ema_model.eval()

        for param in self.ema_model.parameters():
            param.requires_grad = False

        self.batch_size = BATCH_SIZE
        self.lr = LEARNING_RATE
        self.max_epoch = MAX_EPOCH

        self.automatic_optimization = False

        self.train_loss_recorder = AvgMeter()
        self.train_loss_list = list()

        self.val_loss_recorder = AvgMeter()
        self.val_loss_list = list()

        self.ema_loss_recorder = AvgMeter()
        self.ema_loss_list = list()

        self.T = self.forward_model.T

        self.sanity_check_counter = 1

    def forward(
        self,
        x,
        t=None,
        ctx=None,
        inference=False,
        use_ema=False,
        cfg_scale=1.0,
    ):
        if inference:
            return self.sample(x, use_ema=use_ema, cfg_scale=cfg_scale)

        assert t is not None, "Variable 't' can not be 'None'."

        x_latent, _ = self.vae_model.encode(x)
        x_latent_noisy, noise = self.forward_model(x_latent, t)
        if use_ema:
            noise_pred = self.ema_model(x_latent_noisy, t, ctx)
        else:
            noise_pred = self.backward_model(x_latent_noisy, t, ctx)

        return F.mse_loss(noise_pred, noise)

    def sample(
        self,
        ctx,
        n_progress=50,
        scale_factor=8,
        use_ema=False,
        cfg_scale=1.0,
    ):
        if ctx is not None:
            ctx = (
                torch.tensor(ctx)
                .unsqueeze(0)
                .to("cuda" if torch.cuda.is_available() else "cpu")
            )

        if use_ema:
            self.ema_model.eval()
        else:
            self.backward_model.eval()

        progress_image = list()

        with torch.no_grad():
            x = torch.randn(
                1, 3, IMAGE_SIZE // scale_factor, IMAGE_SIZE // scale_factor
            ).to(self.device)
            torch.nn.init.uniform_(x, -1.0, 1.0)

            progress_image.append(x.detach())

            for i in tqdm(reversed(range(1, self.T)), position=0):
                t = (torch.ones(1) * i).long().to(self.device)

                if use_ema:
                    noise_pred = self.ema_model(x, t, ctx)
                else:
                    noise_pred = self.backward_model(x, t, ctx)

                if ctx is not None:
                    if use_ema:
                        uncond_noise_pred = self.ema_model(x, t, None)
                    else:
                        uncond_noise_pred = self.backward_model(x, t, None)

                    noise_pred = torch.lerp(
                        uncond_noise_pred,
                        noise_pred,
                        cfg_scale,
                    )

                alpha_hat = self.forward_model.alpha_hat[t][:, None, None, None]

                alpha = self.forward_model.alpha[t][:, None, None, None]
                beta = self.forward_model.beta[t][:, None, None, None]

                if i > 1:
                    noise = torch.randn_like(x)
                    torch.nn.init.uniform_(noise, -1.0, 1.0)
                else:
                    noise = torch.zeros_like(x)

                coef = beta / torch.sqrt(1.0 - alpha_hat)
                x = (x - coef * noise_pred) / torch.sqrt(alpha)

                variance = torch.sqrt(beta)
                x = x + variance * noise

                x = torch.clamp(x, -1.0, 1.0)

                if (i - 1) % (self.T // n_progress) == 0:
                    progress_image.append(x.detach())

        progress_image_decoded = [
            self.vae_model.decode(img).detach().cpu() for img in progress_image
        ]

        return progress_image, progress_image_decoded

    def training_step(self, batch, batch_idx):
        x, ctx = batch

        if np.random.random() < 0.2:
            ctx = None
        else:
            ctx = ctx.view(-1)

        t = torch.randint(
            1,
            self.T,
            (self.batch_size["train"],),
            device=self.device,
        ).long()

        loss = self(x, t, ctx)

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss, retain_graph=True)
        opt.step()

        self.ema.step_ema(self.ema_model, self.backward_model)

        self.log("train_loss", loss, prog_bar=True)
        self.train_loss_recorder.update(loss.data)

    def on_train_epoch_end(self):
        mean = self.train_loss_recorder.show()
        self.train_loss_list.append(mean.data.cpu().numpy())
        self.train_loss_recorder = AvgMeter()

        self._plot_evaluation_metrics()

    def validation_step(self, batch, batch_idx):
        x, ctx = batch

        ctx = ctx.view(-1)

        t = torch.randint(
            1,
            self.T,
            (self.batch_size["val"],),
            device=self.device,
        ).long()

        loss = self(x, t, ctx)

        ema_loss = self(x, t, ctx, use_ema=True)

        if self.sanity_check_counter == 0:
            self.log("val_loss", loss, prog_bar=True)
            self.log("ema_loss", ema_loss, prog_bar=True)
            self.val_loss_recorder.update(loss.data)
            self.ema_loss_recorder.update(ema_loss.data)

    def on_validation_epoch_end(self):
        if self.sanity_check_counter == 0:
            mean = self.val_loss_recorder.show()
            self.val_loss_list.append(mean.data.cpu().numpy())
            self.val_loss_recorder = AvgMeter()

            mean = self.ema_loss_recorder.show()
            self.ema_loss_list.append(mean.data.cpu().numpy())
            self.ema_loss_recorder = AvgMeter()
        else:
            self.sanity_check_counter -= 1

    def _plot_evaluation_metrics(self):
        loss_img_file = os.path.join(self.run_dir, "LDM_loss_plot.png")
        plt.plot(self.train_loss_list, color="r", label="train")
        plt.plot(self.val_loss_list, color="b", label="validation")
        plt.plot(self.ema_loss_list, color="g", label="ema")
        plt.title("Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.savefig(loss_img_file)
        plt.clf()

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.backward_model.parameters(),
            lr=self.lr,
        )
        return optimizer
