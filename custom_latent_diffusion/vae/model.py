import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
import matplotlib.pyplot as plt
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

from latent_diffusion.config import BATCH_SIZE, LEARNING_RATE, MAX_EPOCH
from latent_diffusion.utils import AvgMeter


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        norm_channel=None,
        norm=True,
        act=True,
        **kwargs,
    ):
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
                bias=not norm,
                **kwargs,
            )
        ]
        if norm:
            assert norm_channel is not None
            layers.append(nn.GroupNorm(norm_channel, out_channels))
        if act:
            layers.append(nn.SiLU())

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class VAEDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_group):
        super().__init__()

        self.block = nn.Sequential(
            ConvNormAct(
                in_channels,
                in_channels,
                kernel_size=3,
                norm_channel=n_group,
            ),
            ConvNormAct(
                in_channels,
                out_channels,
                kernel_size=3,
                norm_channel=n_group,
            ),
            Residual(
                ConvNormAct(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    norm_channel=n_group,
                )
            ),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
        )

    def forward(self, x):
        return self.block(x)


class VAEUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_group):
        super().__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                in_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            ConvNormAct(
                in_channels,
                in_channels,
                kernel_size=3,
                norm_channel=n_group,
            ),
            ConvNormAct(
                in_channels,
                out_channels,
                kernel_size=3,
                norm_channel=n_group,
            ),
            Residual(
                ConvNormAct(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    norm_channel=n_group,
                )
            ),
        )

    def forward(self, x):
        return self.block(x)


class VQVAE(nn.Module):
    def __init__(
        self,
        image_channel=3,
        base_channel=16,
        latent_dim=3,
        codebook_size=4096,
    ):
        super().__init__()

        ####### Vector Quantized
        self.codebook = torch.Tensor(codebook_size, latent_dim).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # This is important for restricting the latent image values range
        self.codebook.uniform_(-1.0, 1.0)

        ####### Encoder
        self.input_conv = ConvNormAct(
            image_channel,
            4 * base_channel,
            kernel_size=3,
            norm_channel=base_channel,
        )
        self.encoder_conv = nn.Sequential(
            VAEDownBlock(4 * base_channel, 4 * base_channel, base_channel),
            VAEDownBlock(4 * base_channel, 8 * base_channel, base_channel),
            VAEDownBlock(8 * base_channel, 16 * base_channel, base_channel),
        )
        self.encoder_latent_proj = nn.Conv2d(
            16 * base_channel,
            latent_dim,
            kernel_size=1,
        )

        ####### Decoder
        self.decoder_latent_proj = nn.Conv2d(
            latent_dim,
            16 * base_channel,
            kernel_size=1,
        )
        self.decoder_conv = nn.Sequential(
            VAEUpBlock(16 * base_channel, 8 * base_channel, base_channel),
            VAEUpBlock(8 * base_channel, 4 * base_channel, base_channel),
            VAEUpBlock(4 * base_channel, 4 * base_channel, base_channel),
        )
        self.output_conv = nn.Conv2d(
            4 * base_channel,
            image_channel,
            kernel_size=3,
            padding=1,
            bias=True,
        )

    def encode(self, x, inference=False):
        x = self.input_conv(x)
        x = self.encoder_conv(x)
        x = self.encoder_latent_proj(x)
        if not inference:
            x, loss = self.quantize(x)
            return x, loss
        else:
            x = self.quantize(x, inference)
            return x

    def quantize(self, x, inference=False):
        B, C, H, W = x.shape

        # B, C, H, W -> B, H, W, C
        x = x.permute(0, 2, 3, 1)

        # B, H, W, C -> B, H*W, C
        x = x.reshape(x.size(0), -1, x.size(-1))

        ################ Find nearest embedding/codebook vector ################

        # dist between (B, H*W, C) and (B, K, C) -> (B, H*W, K)
        dist = torch.cdist(
            x,
            self.codebook[None, :].repeat((x.size(0), 1, 1)),
        )

        # (B, H*W)
        min_encoding_indices = torch.argmin(dist, dim=-1)

        ############# Replace encoder output with nearest codebook #############

        # x -> B*H*W, C
        x = x.reshape((-1, x.size(-1)))

        # quant_out -> B*H*W, C
        quant_out = torch.index_select(
            self.codebook,
            dim=0,
            index=min_encoding_indices.view(-1),
        )

        ############################ Calculate Loss ############################

        if not inference:
            commmitment_loss = torch.mean((quant_out.detach() - x) ** 2)
            codebook_loss = torch.mean((quant_out - x.detach()) ** 2)
            quantize_loss = codebook_loss + commmitment_loss

        ############################### Estimate ###############################

        quant_out = x + (quant_out - x).detach()

        # quant_out -> B, C, H, W
        quant_out = quant_out.reshape((B, H, W, C)).permute(0, 3, 1, 2)

        if not inference:
            return quant_out, quantize_loss

        return quant_out

    def decode(self, x):
        x = self.decoder_latent_proj(x)
        x = self.decoder_conv(x)
        x = self.output_conv(x)
        x = F.hardtanh(x)
        return x

    def forward(self, x, inference=False):
        if not inference:
            z, loss = self.encode(x)
        else:
            z = self.encode(x)

        x = self.decode(z)

        if not inference:
            return x, loss

        return x


class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.reconstruction_loss = nn.MSELoss()

    def forward(self, x, x_hat, vq_loss):
        # Reconstruction Loss
        recon_loss = self.reconstruction_loss(x_hat, x)

        return recon_loss + vq_loss  # VQ Loss


class VAEWrapper(L.LightningModule):
    def __init__(self, run_dir=None):
        super().__init__()

        self.run_dir = run_dir
        self.model = VQVAE()
        self.loss = VAELoss()

        self.batch_size = BATCH_SIZE
        self.lr = LEARNING_RATE
        self.max_epoch = MAX_EPOCH

        # self.train_lpips = LearnedPerceptualImagePatchSimilarity()
        # self.val_lpips = LearnedPerceptualImagePatchSimilarity()
        # self.test_lpips = LearnedPerceptualImagePatchSimilarity()

        # self.train_lpips.eval()
        # self.val_lpips.eval()
        # self.test_lpips.eval()

        # self.train_lpips_recorder = AvgMeter()
        # self.val_lpips_recorder = AvgMeter()

        # self.train_lpips_list = list()
        # self.val_lpips_list = list()

        self.train_loss_recorder = AvgMeter()
        self.val_loss_recorder = AvgMeter()

        self.train_loss = list()
        self.val_loss = list()

        self.sanity_check_counter = 1

        self.automatic_optimization = False

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch

        x_hat, vq_loss = self(x)
        loss = self.loss(x, x_hat, vq_loss)

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        self.log("train_loss", loss, prog_bar=True)
        self.train_loss_recorder.update(loss.data)

        # self.train_lpips.update(
        #     x_hat.repeat(1, 3, 1, 1),
        #     x.repeat(1, 3, 1, 1),
        # )
        # lpips = self.train_lpips.compute().data.cpu()
        # self.train_lpips_recorder.update(lpips.data)
        # self.log("train_lpips", lpips, prog_bar=True)

    def on_train_epoch_end(self):
        mean = self.train_loss_recorder.show()
        self.train_loss.append(mean.data.cpu().numpy())
        self.train_loss_recorder = AvgMeter()

        # mean = self.train_lpips_recorder.show()
        # self.train_lpips_list.append(mean.data.cpu().numpy())
        # self.train_lpips_recorder = AvgMeter()

        self._plot_evaluation_metrics()

    def validation_step(self, batch, batch_idx):
        x, _ = batch

        x_hat, vq_loss = self(x)
        loss = self.loss(x, x_hat, vq_loss)

        if self.sanity_check_counter == 0:
            self.log("val_loss", loss, prog_bar=True)
            self.val_loss_recorder.update(loss.data)

            # self.val_lpips.update(
            #     x_hat.repeat(1, 3, 1, 1),
            #     x.repeat(1, 3, 1, 1),
            # )
            # lpips = self.val_lpips.compute().data.cpu()

            # self.log("val_lpips", lpips, prog_bar=True)

            # self.val_lpips_recorder.update(lpips.data)

    def on_validation_epoch_end(self):
        if self.sanity_check_counter == 0:
            mean = self.val_loss_recorder.show()
            self.val_loss.append(mean.data.cpu().numpy())
            self.val_loss_recorder = AvgMeter()

            # mean = self.val_lpips_recorder.show()
            # self.val_lpips_list.append(mean.data.cpu().numpy())
            # self.val_lpips_recorder = AvgMeter()
        else:
            self.sanity_check_counter -= 1

    def test_step(self, batch, batch_idx):
        x, _ = batch

        x_hat, vq_loss = self(x)
        loss = self.loss(x, x_hat, vq_loss)

        self.log("test_loss", loss, prog_bar=True, logger=True)

        # self.test_lpips.update(
        #     x_hat.repeat(1, 3, 1, 1),
        #     x.repeat(1, 3, 1, 1),
        # )

        # self.log("test_lpips", self.test_lpips.compute(), prog_bar=True, logger=True)

    def _plot_evaluation_metrics(self):
        # VAE Loss
        vae_loss_img_file = os.path.join(self.run_dir, "VAE_loss_plot.png")
        plt.plot(self.train_loss, color="r", label="train")
        plt.plot(self.val_loss, color="b", label="validation")
        plt.title("VAE Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.savefig(vae_loss_img_file)
        plt.clf()

        # LPIPS
        # lpips_img_file = os.path.join(self.run_dir, "VAE_lpips_plot.png")
        # plt.plot(self.train_lpips_list, color="r", label="train")
        # plt.plot(self.val_lpips_list, color="b", label="validation")
        # plt.title("LPIPS Curves")
        # plt.xlabel("Epoch")
        # plt.ylabel("LPIPS")
        # plt.legend()
        # plt.grid()
        # plt.savefig(lpips_img_file)
        # plt.clf()

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
        )

        return optimizer
