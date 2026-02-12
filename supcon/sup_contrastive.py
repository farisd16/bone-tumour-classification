"""
Supervised Contrastive Learning Components.

This module contains:

1. SupConLoss:
   Implementation of the Supervised Contrastive Loss introduced in
   "Supervised Contrastive Learning" (Khosla et al., 2020).

2. TwoViewDataset:
   A dataset wrapper that generates two independently augmented views
   of the same image, required for contrastive learning.

3. SupConModel:
   A model consisting of an encoder backbone and an MLP projection head,
   used for supervised contrastive representation learning.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F



# Supervised Contrastive Loss (SupConLoss)
class SupConLoss(nn.Module):
    """Supervised Contrastive Loss from 'Supervised Contrastive Learning' (Khosla et al., 2020)."""
    def __init__(self, temperature=0.1, contrast_mode='all', base_temperature=0.07):
        super().__init__()
        self.temperature = temperature      # controls how much the model cares about similarity differences
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: tensor of shape [batch_size, n_views, feature_dim]
            labels: tensor of shape [batch_size]
        """
        device = features.device

        features = F.normalize(features, p=2, dim=2)  # normalize embeddings
        batch_size = features.shape[0]

        # Creates the mask to know which anchors are treated as positives
        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )

        logits_max = torch.max(anchor_dot_contrast, dim=1, keepdim=True)[0]
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, anchor_count)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size * anchor_count).to(device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        mask_sum = mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask_sum + 1e-9)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss



# TwoViewDataset (needed for SupCon training)

class TwoViewDataset(torch.utils.data.Dataset):
    """
    Wraps any dataset and returns TWO randomly augmented views per image.
    Required for contrastive learning.
    """
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        view1 = self.transform(img)
        view2 = self.transform(img)
        return torch.stack([view1, view2], dim=0), label



# SupCon Model (Encoder + Projection Head)
class SupConModel(nn.Module):
    def __init__(self, encoder, feature_dim=128):
        super().__init__()
        self.encoder = encoder

        # MLP projection head
        self.projector = nn.Sequential(
            nn.Linear(512, 512),   # for ResNet34: encoder output is 512-D
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim)
        )

    def forward(self, x):
        h = self.encoder(x)                 # encoder output (batch, 512)
        z = self.projector(h)               # projection head (batch, feature_dim)
        return F.normalize(z, dim=1)        # normalized embeddings
