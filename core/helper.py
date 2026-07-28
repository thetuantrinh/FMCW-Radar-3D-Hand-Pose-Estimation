# helper.py
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch


def split_dataset(
    dataset,
    batch_size,
    train_ratio=0.8,
    shuffle=True,
    seed=42,
    workers=4,
    pin_memory=True,
):
    dataset_size = len(dataset)
    indices = np.arange(dataset_size)
    if shuffle:
        np.random.seed(seed=seed)
        np.random.shuffle(indices)
        
    split = int(np.floor(train_ratio * dataset_size))
    train_indices, val_indices = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


def create_batch_loader(
    data,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=4,
):
    # If num_workers is 0, prefetch_factor must be set to None in PyTorch
    pf = prefetch_factor if num_workers > 0 else None
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=pf,
        pin_memory=pin_memory,
    )