"""
Dataset loading for MLLM-driven source-free domain adaptation.

Supports loading pseudo-labeled data from MLLM voting strategies,
with optional individual teacher votes for curriculum learning.
"""

import os
import torch
import torchvision.transforms as transforms
from PIL import ImageFile, Image
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def make_dataset(image_list, labels):
    """Parse an image list file into (path, label) or (path, [labels]) tuples."""
    if labels:
        return [(image_list[i].strip(), labels[i, :]) for i in range(len(image_list))]

    images = []
    for val in image_list:
        parts = val.split()
        if len(parts) == 2:
            # Standard: path label
            images.append((parts[0], int(parts[1])))
        elif len(parts) == 3:
            # path label indicator — keep only path and label
            images.append((parts[0], int(parts[1])))
        else:
            # Multiple labels per image (individual MLLM votes)
            images.append((parts[0], np.array([int(la) for la in parts[1:]])))
    return images


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class ImageList_idx(torch.utils.data.Dataset):
    """Dataset from a text file listing (image_path, label) pairs."""

    def __init__(self, image_list, labels=None, transform=None, mode='RGB'):
        self.imgs = make_dataset(image_list, labels)
        if len(self.imgs) == 0:
            raise RuntimeError("Found 0 images in the provided image list.")
        self.transform = transform
        self.loader = rgb_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgs)


class ImageList_idx_ind_vote(torch.utils.data.Dataset):
    """Dataset with individual MLLM vote labels (3 teachers)."""

    def __init__(self, image_list, labels=None, transform=None, mode='RGB'):
        self.imgs = make_dataset(image_list, labels)
        if len(self.imgs) == 0:
            raise RuntimeError("Found 0 images in the provided image list.")
        self.transform = transform
        self.loader = rgb_loader

    def __getitem__(self, index):
        path, (teacher1, teacher2, teacher3) = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, teacher1, teacher2, teacher3

    def __len__(self):
        return len(self.imgs)


class ImageList_idx_ind_vote_ssl(torch.utils.data.Dataset):
    """Dataset with individual MLLM vote labels (3 teachers) + weak/strong augmentations for SSL."""

    def __init__(self, image_list, labels=None, weak_transform=None, strong_transform=None, mode='RGB'):
        self.imgs = make_dataset(image_list, labels)
        if len(self.imgs) == 0:
            raise RuntimeError("Found 0 images in the provided image list.")
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        self.loader = rgb_loader

    def __getitem__(self, index):
        path, (teacher1, teacher2, teacher3) = self.imgs[index]
        img = self.loader(path)
        weak_img = self.weak_transform(img) if self.weak_transform else img
        strong_img = self.strong_transform(img) if self.strong_transform else img
        return weak_img, strong_img, teacher1, teacher2, teacher3

    def __len__(self):
        return len(self.imgs)


class ImageList_idx_ind_vote_ssl_4models(torch.utils.data.Dataset):
    """Dataset with individual MLLM vote labels (4 teachers) + weak/strong augmentations for SSL."""

    def __init__(self, image_list, labels=None, weak_transform=None, strong_transform=None, mode='RGB'):
        self.imgs = make_dataset(image_list, labels)
        if len(self.imgs) == 0:
            raise RuntimeError("Found 0 images in the provided image list.")
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        self.loader = rgb_loader

    def __getitem__(self, index):
        path, (teacher1, teacher2, teacher3, teacher4) = self.imgs[index]
        img = self.loader(path)
        weak_img = self.weak_transform(img) if self.weak_transform else img
        strong_img = self.strong_transform(img) if self.strong_transform else img
        return weak_img, strong_img, teacher1, teacher2, teacher3, teacher4

    def __len__(self):
        return len(self.imgs)


# ---------------------------------------------------------------------------
# Split dataset wrappers
# ---------------------------------------------------------------------------

class _SplitDataset(torch.utils.data.Dataset):
    def __init__(self, underlying_dataset, keys):
        super().__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
        self.transform = None

    def __getitem__(self, key):
        x, y = self.underlying_dataset[self.keys[key]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.keys)


class _SplitDataset_ind_vote(torch.utils.data.Dataset):
    def __init__(self, underlying_dataset, keys):
        super().__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
        self.transform = None

    def __getitem__(self, key):
        x, y1, y2, y3 = self.underlying_dataset[self.keys[key]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y1, y2, y3

    def __len__(self):
        return len(self.keys)


class _SplitDataset_ind_vote_ssl(torch.utils.data.Dataset):
    def __init__(self, underlying_dataset, keys):
        super().__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys

    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]

    def __len__(self):
        return len(self.keys)


class _SplitDataset_ind_vote_ssl_4mllms(torch.utils.data.Dataset):
    def __init__(self, underlying_dataset, keys):
        super().__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys

    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]

    def __len__(self):
        return len(self.keys)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transform(mode, type_='default'):
    if mode in ['val', 'test']:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    elif mode == 'train':
        if type_ == 'default':
            return transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        elif 'randaug' in type_:
            try:
                _, num_ops, magnitude = type_.split('_')
            except ValueError:
                num_ops, magnitude = 3, 10
            return transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=int(num_ops), magnitude=int(magnitude)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            raise ValueError(f"Unknown transform type: {type_}")
    else:
        raise ValueError(f"Unknown transform mode: {mode}")


# ---------------------------------------------------------------------------
# Main dataset construction
# ---------------------------------------------------------------------------

def _get_all_indices(dataset):
    """Return indices for all samples (use full dataset for adaptation)."""
    return list(range(len(dataset)))


def get_target_dataset(args):
    """Build train / validation / test datasets for the target domain.

    Returns:
        (train_dataset, val_dataset, test_dataset, target_domain_name)
    """
    data_path = os.path.join(args.data_dir, args.dataset)
    domains = sorted([f.name for f in os.scandir(data_path) if f.is_dir()])

    src_domain = domains[args.source]
    tgt_domain = domains[args.target]
    print(f"Source Domain : {src_domain}")
    print(f"Target Domain : {tgt_domain}")

    mllm_strategies = ['majority_vote', 'all_agree_voting',
                       'majority_vote_no_disagree', 'all_disagree']

    if args.adapt is not None and 'mllm' in args.adapt and args.mllm_strategy in mllm_strategies:
        # --- MLLM pseudo-label path (main paper method) ---
        mllm_dir = f'{args.data_dir}/{args.dataset}_{args.adapt}'

        # Training data (pseudo-labeled by MLLMs)
        if args.ind_vote:
            txt_train = open(f'{mllm_dir}/{args.mllm_strategy}_ind_vote_{tgt_domain}.txt').readlines()
            if args.ssl == 'fixmatch' and 'blip2_llava34b_sharegpt13b_instructblip' in args.adapt:
                target_dataset = ImageList_idx_ind_vote_ssl_4models(txt_train)
            elif args.ssl == 'fixmatch':
                target_dataset = ImageList_idx_ind_vote_ssl(txt_train)
            else:
                target_dataset = ImageList_idx_ind_vote(txt_train)
        else:
            txt_train = open(f'{mllm_dir}/{args.mllm_strategy}_{tgt_domain}.txt').readlines()
            target_dataset = ImageList_idx(txt_train)

        print(f"Training set size: {len(target_dataset)}")

        # Test data (ground truth labels)
        txt_test = open(f'{args.data_dir}/{args.dataset}/{tgt_domain}_new_key.txt').readlines()
        test_dataset = ImageList_idx(txt_test)
        print(f"Test set size: {len(test_dataset)}")

        # Validation splits by agreement category
        val_datasets = []
        for split_name in ['all_disagree_gt', 'all_agree_gt', 'only_two_agree_gt', 'incorrect_vote_gt']:
            txt_val = open(f'{mllm_dir}/{split_name}_{tgt_domain}.txt').readlines()
            val_datasets.append(ImageList_idx(txt_val))
            print(f"Val split ({split_name}): {len(val_datasets[-1])}")

    elif args.adapt is None or args.adapt == 'all':
        # --- No-adapt baseline ---
        txt_test = open(f'{args.data_dir}/{args.dataset}/{tgt_domain}_new_key.txt').readlines()
        target_dataset = ImageList_idx(txt_test)
        test_dataset = ImageList_idx(txt_test)
        val_datasets = None
    else:
        raise ValueError(f"Unknown adaptation strategy: {args.adapt}")

    # Build split datasets
    tr_idx = _get_all_indices(target_dataset)
    te_idx = _get_all_indices(test_dataset)

    if args.ind_vote and args.ssl is None:
        train_dataset = _SplitDataset_ind_vote(target_dataset, tr_idx)
    elif args.ind_vote and args.ssl == 'fixmatch' and 'blip2_llava34b_sharegpt13b_instructblip' in (args.adapt or ''):
        train_dataset = _SplitDataset_ind_vote_ssl_4mllms(target_dataset, tr_idx)
    elif args.ind_vote and args.ssl == 'fixmatch':
        train_dataset = _SplitDataset_ind_vote_ssl(target_dataset, tr_idx)
    else:
        train_dataset = _SplitDataset(target_dataset, tr_idx)

    test_dataset = _SplitDataset(test_dataset, te_idx)

    # Apply transforms
    if args.ssl == 'fixmatch':
        train_dataset.weak_transform = get_transform(mode='train', type_='default')
        train_dataset.strong_transform = get_transform(mode='train', type_='randaug_3_10')
    else:
        train_dataset.transform = get_transform(mode='train', type_=args.aug)
    test_dataset.transform = get_transform(mode='test')

    # Validation transforms
    if val_datasets is not None:
        val_idx = [_get_all_indices(vd) for vd in val_datasets]
        val_datasets = [_SplitDataset(vd, idx) for vd, idx in zip(val_datasets, val_idx)]
        for vd in val_datasets:
            vd.transform = get_transform(mode='val')
        val_dataset = val_datasets
    else:
        val_dataset = test_dataset

    # Handle train data smaller than batch size
    if len(train_dataset) < args.batch_size:
        multiplier = int(args.batch_size / len(train_dataset)) + 1
        train_dataset.keys = train_dataset.keys * multiplier

    return train_dataset, val_dataset, test_dataset, tgt_domain
