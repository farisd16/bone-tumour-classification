import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.scores = []

    def update(self, val):
        self.scores.append(val)

    def show(self):
        out = torch.mean(
            torch.stack(self.scores[np.maximum(len(self.scores) - self.num, 0) :])
        )
        return out


class EMA(object):
    def __init__(self, beta=0.9):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new

        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=1024):
        self.step += 1

        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            return None

        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


def animate_result(
    image_tensors,
    caption,
    run_name,
    class_name,
):
    """
    Creates a GIF from a list of PyTorch image tensors
    with a caption.

    Args:
      image_tensors: A list of PyTorch image tensors.
                     Assumed to be in CHW format,
                     and values normalized to [-1, 1].
      caption: The caption to display below the GIF.
      output_filename: The filename for the output GIF.
    """
    RUN_DIR = f"latent_diffusion/diffusion/runs/{run_name}"

    fig, ax = plt.subplots(
        gridspec_kw={
            "wspace": 0,
            "hspace": 2,
            "right": 1,
            "left": 0,
        }
    )

    # Remove axes ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    imgs = list()
    for idx, image_tensor in enumerate(image_tensors):
        # Assuming the tensor is in CHW format.
        assert len(image_tensor.shape) == 4, "Expected 4D tensor"

        # Convert to HWC and to numpy.
        if image_tensor.shape[1] == 3:
            image_np = (
                F.interpolate(
                    image_tensor,
                    scale_factor=4,
                )
                .squeeze(0)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
        else:
            image_np = (
                image_tensor.repeat(1, 3, 1, 1)
                .squeeze(0)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )

        # Scale to 0-255 range and convert to uint8.
        image_np = (image_np + 1) / 2  # Rescale to 0-1
        image_np = (image_np * 255).astype("uint8")

        if idx == len(image_tensors) - 1:
            for _ in range(32):
                img = ax.imshow(image_np)
                imgs.append([img])
        else:
            img = ax.imshow(image_np, aspect="auto")
            imgs.append([img])

    plt.tight_layout()

    # Add caption
    plt.figtext(
        0.5,
        0.05,
        caption,
        wrap=True,
        horizontalalignment="center",
        fontsize=16,
    )

    anim = animation.ArtistAnimation(
        fig,
        imgs,
        interval=256,
        blit=True,
        repeat_delay=1000,
    )
    anim.save(f"{RUN_DIR}/{class_name}_animation.gif")

    plt.close(fig)
