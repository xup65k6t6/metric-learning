import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models


class EmbeddingModel(nn.Module):
    def __init__(self, num_classes=4, embedding_dim=128, unfreeze_layers=False):
        super().__init__()
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        if not unfreeze_layers:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Modify backbone to remove original classification head
        num_ftrs = self.backbone.classifier[1].in_features

        # Custom embedding head
        self.embedding_head = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x):
        # Extract features
        features = self.backbone.features(x)
        pooled = self.backbone.avgpool(features)
        pooled = torch.flatten(pooled, 1)

        # Generate embedding
        embedding = self.embedding_head(pooled)

        # L2 normalize the embedding
        return F.normalize(embedding, p=2, dim=1)


# bypass mps NotImplementedError: The operator 'aten::_cdist_backward' is not currently implemented for the MPS device.
def pairwise_euclidean_distance(x):
    """Compute pairwise distances without torch.cdist."""
    x_square = torch.sum(x**2, dim=1, keepdim=True)  # Shape: (N, 1)
    distances = x_square + x_square.T - 2 * torch.mm(x, x.T)  # (N, N) matrix
    distances = torch.sqrt(
        torch.relu(distances)
    )  # Clamp to avoid small negative values from numerical errors
    return distances


class LiftedStructureLoss(nn.Module):
    def __init__(self, margin=1.0, top_k=None):
        super().__init__()
        self.margin = margin
        self.top_k = top_k

    def forward(self, embeddings, labels):
        distances = torch.cdist(embeddings, embeddings, p=2.0)
        # Positive and negative masks
        labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
        diag_mask = ~torch.eye(
            labels_matrix.shape[0], dtype=torch.bool, device=labels.device
        )
        positive_mask = labels_matrix & diag_mask
        negative_mask = ~labels_matrix & diag_mask

        # positive pairs indices
        pos_pairs = torch.where(positive_mask)
        if len(pos_pairs[0]) == 0:  # Handle edge case
            return torch.tensor(0.0, device=embeddings.device)

        losses = []
        for i, j in zip(pos_pairs[0], pos_pairs[1]):
            neg_i = distances[i][negative_mask[i]]
            sum_exp_i = torch.sum(torch.exp(self.margin - neg_i))

            neg_j = distances[j][negative_mask[j]]
            sum_exp_j = torch.sum(torch.exp(self.margin - neg_j))

            # Compute L_ij
            L_ij = torch.log(sum_exp_i + sum_exp_j) + distances[i, j]
            losses.append(torch.pow(torch.relu(L_ij), 2))

        all_losses = torch.stack(losses)

        # Apply top-k selection if enabled
        if self.top_k is not None:
            num_losses = all_losses.size(0)
            k = min(max(10, int(num_losses * self.top_k)), num_losses)
            hardest_losses = torch.topk(all_losses, k).values
            return torch.mean(hardest_losses) / 2

        return torch.mean(torch.stack(losses)) / 2


from sklearn.model_selection import train_test_split


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path="best_model.pt"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.best_state = None

    def __call__(self, val_loss, model, device):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, device)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, device)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, device):
        """Save model checkpoint and keep best state in memory."""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ..."
            )

        self.best_state = model.state_dict().copy()
        torch.save(self.best_state, self.path)
        self.val_loss_min = val_loss

    def load_best_model(self, model):
        """Load the best model state from memory."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        return model


def compute_distances(embeddings, labels):
    # Compute class centers
    class_centers = {}
    for cls in torch.unique(labels):
        cls = cls.item()  # tensor to int
        class_mask = labels == cls
        class_centers[cls] = embeddings[class_mask].mean(dim=0)

    # Compute inter-class distances
    inter_class_distances = []
    for cls1, center1 in class_centers.items():
        for cls2, center2 in class_centers.items():
            if cls1 != cls2:
                inter_class_distances.append(torch.norm(center1 - center2).item())

    # Compute intra-class distances
    intra_class_distances = {}
    total_intra_distances = []
    for cls in torch.unique(labels):
        cls = cls.item()
        class_mask = labels == cls
        class_center = class_centers[cls]
        distances = torch.norm(embeddings[class_mask] - class_center, dim=1)
        intra_class_distances[f"label:{cls}"] = distances.mean().item()
        total_intra_distances.extend(distances.tolist())

    return {
        "avg_inter_class_distance": np.mean(inter_class_distances),
        "avg_intra_class_distance": np.mean(total_intra_distances),
    } | intra_class_distances


def split_df(df, train_size=0.8, seed=42):
    """
    Split DataFrame into train and validation sets while maintaining class distribution

    Args:
        df: DataFrame containing image paths and labels
        train_size: Fraction of data to use for training
        seed: Random seed for reproducibility

    Returns:
        train_df: DataFrame for training
        val_df: DataFrame for validation
    """
    unique_labels = df["label"].unique()

    train_df = pd.DataFrame()
    val_df = pd.DataFrame()

    # Split each class separately to maintain distribution
    for label in unique_labels:
        class_df = df[df["label"] == label]

        train_class, val_class = train_test_split(
            class_df, train_size=train_size, random_state=seed, shuffle=True
        )

        train_df = pd.concat([train_df, train_class])
        val_df = pd.concat([val_df, val_class])

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    return train_df, val_df


def setup_scheduler(optimizer, epochs, lrf, warmup_epochs):
    """Initialize training learning rate scheduler.

    lrf: final learning rate factor
    warmup_epochs: number of epochs for warm-up phase
    """

    # Lambda function for linear warm-up and then linear decay
    def lr_lambda(x):
        if x < warmup_epochs:
            # Linear warm-up phase
            return (1.0 - lrf) * (x / warmup_epochs) + lrf
        else:
            # Linear decay phase after warm-up
            return max(1 - (x - warmup_epochs) / (epochs - warmup_epochs), lrf)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler


def evaluate_embeddings_strctured(model, dataloader, device):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for images, anchor_label in dataloader:
            images = images.to(device)
            batch_embeddings = model(images)

            embeddings.append(batch_embeddings.cpu())
            labels.append(anchor_label)

    embeddings = torch.cat(embeddings)
    labels = torch.cat(labels)

    return compute_distances(embeddings, labels)
