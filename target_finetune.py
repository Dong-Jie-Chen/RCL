"""
Main training script for RCL: three-stage MLLM-driven curriculum learning
for source-free domain adaptation.

Stage 1 – RKT (Reliable Knowledge Transfer):
  Train on fully-agreed MLLM pseudo-labels (all-agree voting).

Stage 2 – SMKE (Self-correcting and MLLM-guided Knowledge Expansion):
  Expand to partially-agreed samples; student self-corrects when confident,
  otherwise falls back to majority-vote MLLM label.

Stage 3 – MMR (Multi-hot Masking Refinement):
  Train on all samples; multi-hot mask constrains the student's logit space
  to the union of MLLM-suggested classes, with consistency regularisation.

Usage (example for Office-Home, source=0, target=1):
  Stage 1 (RKT):
    python target_finetune.py --dataset office_home --pretrain SHOT --adapt mllm_clip_llava7b_sharegpt7b \
      --mllm_strategy all_agree_voting --adapt_step 3000 --lr 1e-4 --batch_size 128 --bAcc \
      --source 0 --target 1 --ckpt_dir ./source_models_from_SHOT/seed2021

  Stage 2 (SMKE):
    python target_finetune.py --dataset office_home --pretrain PREV_STAGE --adapt mllm_clip_llava7b_sharegpt7b \
      --mllm_strategy majority_vote_no_disagree --cl curriculum_2and3_self_correct --p_th 0.7 \
      --adapt_step 5000 --lr 1e-5 --batch_size 128 --bAcc --source 0 --target 1 \
      --ckpt_dir ./logs/office_home/<stage1_work_dir>

  Stage 3 (MMR):
    python target_finetune.py --dataset office_home --pretrain PREV_STAGE --adapt mllm_clip_llava7b_sharegpt7b \
      --mllm_strategy majority_vote --cl curriculum_teacher_class_filter_conf_th_ssl --p_th 0.9 \
      --ind_vote --ssl fixmatch --adapt_step 8000 --lr 1e-5 --batch_size 128 --bAcc \
      --source 0 --target 1 --ckpt_dir ./logs/office_home/<stage2_work_dir>
"""

import argparse
import os
import json
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import OrderedDict
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

from model import SHOT
from losses import CrossEntropyLabelSmooth
from dataset import get_target_dataset
from data_loader import InfiniteDataLoader
from curriculum_learning import (
    curriculum_2and3_self_correct,
    curriculum_teacher_class_filter_conf_th_ssl,
    curriculum_teacher_class_filter_conf_th_ssl_4mllms,
)
from ssl_utils import train_one_step_fixmatch
import utils

warnings.filterwarnings(action='ignore')


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='RCL: MLLM-driven Curriculum Learning for SFDA')

    # Data
    parser.add_argument('--data_dir', default='./data', type=str)
    parser.add_argument('--dataset', default='office_home', type=str,
                        choices=['office_home', 'visda', 'domainnet'])
    parser.add_argument('--source', default=None, type=int, help='Source domain index')
    parser.add_argument('--target', default=None, type=int, help='Target domain index')

    # Source model
    parser.add_argument('--pretrain', default='SHOT', type=str,
                        choices=['SHOT', 'PREV_STAGE'],
                        help='SHOT = load from SHOT source model; PREV_STAGE = load from a previous RCL stage')
    parser.add_argument('--ckpt_dir', default=None, type=str,
                        help='Root directory for source / previous-stage checkpoints')

    # Adaptation strategy
    parser.add_argument('--adapt', default=None, type=str,
                        choices=[None, 'all',
                                 'mllm_clip_llava7b_sharegpt7b',
                                 'mllm_blip2_llava34b_sharegpt13b_instructblip'],
                        help='MLLM combination for pseudo-label generation')
    parser.add_argument('--mllm_strategy', default=None, type=str,
                        choices=['all_agree_voting', 'majority_vote',
                                 'majority_vote_no_disagree', 'all_disagree'],
                        help='Voting strategy for MLLM pseudo-labels')

    # Curriculum learning
    parser.add_argument('--cl', default=None, type=str,
                        choices=['curriculum_2and3_self_correct',
                                 'curriculum_teacher_class_filter_conf_th_ssl',
                                 'curriculum_teacher_class_filter_conf_th_ssl_4mllms'],
                        help='Curriculum learning strategy (Stage 2 or 3)')
    parser.add_argument('--p_th', default=None, type=float,
                        help='Confidence threshold for curriculum filtering')
    parser.add_argument('--ind_vote', action='store_true', default=False,
                        help='Use individual per-MLLM vote labels (for Stage 3)')

    # Semi-supervised learning
    parser.add_argument('--ssl', default=None, type=str, choices=['fixmatch'],
                        help='Semi-supervised method (Stage 3)')
    parser.add_argument('--lambda_u', default=0.5, type=float,
                        help='Weight for unlabeled loss in FixMatch')

    # Training
    parser.add_argument('--work_dir', default='./result', type=str)
    parser.add_argument('--adapt_step', default=1000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--optim', default='adam', type=str, choices=['adam', 'sgd'])
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--wd', default=0, type=float, help='Weight decay')
    parser.add_argument('--scheduler', default=None, type=str, choices=['cosine', 'step'])
    parser.add_argument('--aug', default='default', type=str,
                        choices=['default', 'randaug'])
    parser.add_argument('--loss', default='ce', type=str, choices=['ce', 'ce_ls'],
                        help='Loss function: ce (cross-entropy) or ce_ls (label smoothing)')
    parser.add_argument('--label_smoothing', default=0, type=float)
    parser.add_argument('--val_freq', default=100, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--bAcc', action='store_true', default=False,
                        help='Use balanced accuracy as selection metric')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def train_one_step(model, criterion, optimizer, x, y, args):
    """One training step with standard optimizer."""
    model.train()
    x, y = x.cuda(), y.cuda()

    output = model(x)
    loss = criterion(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    acc, _ = utils.accuracy(output, y, topk=(1, 5))
    return loss.item(), acc


def test(model, criterion, loader, flag=False, bAcc=False, return_label=False):
    """Evaluate model on a data loader."""
    model.eval()
    test_loss = utils.AverageMeter()
    test_acc = utils.AverageMeter()
    test_confidence = utils.AverageMeter()

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.cuda(), y.cuda()
            output = model(x)
            loss = criterion(output, y)
            conf = F.softmax(output, dim=1).max(1)[0]

            if flag or bAcc:
                preds = torch.argmax(output.detach().cpu(), 1)
                labels = y.detach().cpu()
                if i == 0:
                    total_pred, total_label = preds, labels
                else:
                    total_pred = torch.cat((total_pred, preds))
                    total_label = torch.cat((total_label, labels))

            acc, _ = utils.accuracy(output, y, topk=(1, 5))
            test_acc.update(acc[0].item(), x.size(0))
            test_loss.update(loss.item(), x.size(0))
            test_confidence.update(conf.mean().item(), x.size(0))

    if flag:
        cm = confusion_matrix(total_label, total_pred)
        per_class_acc = (cm.diagonal() / cm.sum(axis=1) * 100).mean()
        return test_loss.avg, test_acc.avg, per_class_acc

    if bAcc:
        bacc = balanced_accuracy_score(total_label.numpy(), total_pred.numpy()) * 100
        if return_label:
            return test_loss.avg, test_acc.avg, bacc, test_confidence.avg, total_label.numpy(), total_pred.numpy()
        return test_loss.avg, test_acc.avg, bacc, test_confidence.avg

    return test_loss.avg, test_acc.avg, test_confidence.avg


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

DOMAIN_DICT = {
    'office_home': ['A', 'C', 'P', 'R'],
    'domainnet': ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'],
    'visda': ['T', 'V'],
}


def get_ckpt(args):
    """Resolve checkpoint path(s) based on pretrain strategy."""
    if args.pretrain == 'SHOT':
        src_abbrev = DOMAIN_DICT[args.dataset][args.source]
        return {
            'netF': os.path.join(args.ckpt_dir, f'{args.dataset}/{src_abbrev}/source_F.pt'),
            'netB': os.path.join(args.ckpt_dir, f'{args.dataset}/{src_abbrev}/source_B.pt'),
            'netC': os.path.join(args.ckpt_dir, f'{args.dataset}/{src_abbrev}/source_C.pt'),
        }
    elif args.pretrain == 'PREV_STAGE':
        return os.path.join(args.ckpt_dir, f'source{args.source}/target{args.target}/test_best_ckpt.pth')
    else:
        raise ValueError(f"Unknown pretrain strategy: {args.pretrain}")


def get_loss(args, num_classes):
    if args.loss == 'ce':
        return nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    elif args.loss == 'ce_ls':
        return CrossEntropyLabelSmooth(num_classes)
    else:
        raise ValueError(f"Unknown loss: {args.loss}")


def get_scheduler(scheduler_name, optimizer):
    if scheduler_name == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    elif scheduler_name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    return None


# ---------------------------------------------------------------------------
# Main adaptation loop
# ---------------------------------------------------------------------------

def train_on_target(args):
    """Run adaptation on the target domain and return accuracy metrics."""
    args.save_path = os.path.join(args.work_dir, f'target{args.target}')
    os.makedirs(args.save_path, exist_ok=True)

    # Data
    train_dataset, val_dataset, test_dataset, target_domain = get_target_dataset(args)

    num_classes = {'office_home': 65, 'domainnet': 126, 'visda': 12}[args.dataset]
    print(f"Number of classes: {num_classes}")

    train_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, num_workers=8)

    if isinstance(val_dataset, list):
        val_loader = [DataLoader(v, batch_size=args.batch_size * 3, shuffle=False, num_workers=8)
                      for v in val_dataset]
    else:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 3, shuffle=False, num_workers=8)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 3, shuffle=False, num_workers=8)

    # Model
    if args.pretrain == 'SHOT':
        model = SHOT(ckpts=args.ckpts, dataset=args.dataset).cuda()
    elif args.pretrain == 'PREV_STAGE':
        model = SHOT(ckpts=None, dataset=args.dataset, prev_stage_ckpt=args.ckpts).cuda()

    # Loss & optimizer
    criterion = get_loss(args, num_classes)

    if args.adapt is not None:
        param = model.parameters()
        if args.optim == 'adam':
            optimizer = torch.optim.Adam(param, lr=args.lr, weight_decay=args.wd)
        else:
            optimizer = torch.optim.SGD(param, lr=args.lr, weight_decay=args.wd,
                                        momentum=0.9, nesterov=True)
        scheduler = get_scheduler(args.scheduler, optimizer)

    # Tracking
    train_loss = utils.AverageMeter()
    train_acc = utils.AverageMeter()
    if args.ssl == 'fixmatch':
        train_ulb_loss = utils.AverageMeter()

    best_acc, best_bacc, selected_acc = 0, 0, 0
    train_iterator = iter(train_loader)

    # --- No-adaptation baseline ---
    if args.adapt is None:
        if args.bAcc:
            tl, ta, tb, tc = test(model, criterion, test_loader, bAcc=True)
            print(f"No adapt  Test Loss: {tl:.6f}  Acc: {ta:.6f}  bAcc: {tb:.6f}")
            return ta, ta, ta, tb
        else:
            tl, ta, tc = test(model, criterion, test_loader)
            print(f"No adapt  Test Loss: {tl:.6f}  Acc: {ta:.6f}")
            return ta, ta, ta, 0

    # --- Adaptation loop ---
    for step in range(args.adapt_step):
        # ---- Curriculum learning ----
        if args.ind_vote:
            if args.cl == 'curriculum_teacher_class_filter_conf_th_ssl':
                x_w, x_s, y1, y2, y3 = next(train_iterator)
                x, ulb_mask, y, _ = curriculum_teacher_class_filter_conf_th_ssl(model, x_w, y1, y2, y3, args)
            elif args.cl == 'curriculum_teacher_class_filter_conf_th_ssl_4mllms':
                x_w, x_s, y1, y2, y3, y4 = next(train_iterator)
                x, ulb_mask, y, _ = curriculum_teacher_class_filter_conf_th_ssl_4mllms(model, x_w, y1, y2, y3, y4, args)
            else:
                raise ValueError(f"When --ind_vote is set, --cl must be one of "
                                 f"curriculum_teacher_class_filter_conf_th_ssl[_4mllms]")
        else:
            x, y = next(train_iterator)
            if args.cl == 'curriculum_2and3_self_correct':
                x, y, _ = curriculum_2and3_self_correct(model, x, y, args)

        # ---- Training step ----
        if args.ssl == 'fixmatch':
            loss, ulb_loss, acc = train_one_step_fixmatch(model, criterion, optimizer, x_w, x_s, ulb_mask, y, args)
            train_ulb_loss.update(ulb_loss, x.size(0))
        else:
            loss, acc = train_one_step(model, criterion, optimizer, x, y, args)
        train_loss.update(loss, x.size(0))
        train_acc.update(acc[0].item(), x.size(0))

        if scheduler:
            scheduler.step()

        # ---- Logging & evaluation ----
        if step % args.val_freq == 0 or step == args.adapt_step - 1:
            result = {
                'train_step': step,
                'train_loss': train_loss.avg,
                'train_acc': train_acc.avg,
            }
            train_loss.reset()
            train_acc.reset()
            if args.ssl == 'fixmatch':
                result['train_ulb_loss'] = train_ulb_loss.avg
                train_ulb_loss.reset()

            # Test on target test set
            test_bacc = 0.0
            if args.dataset == 'visda':
                tl, ta, tpc = test(model, criterion, test_loader, bAcc=True, flag=True)
                test_bacc = tpc
            elif args.bAcc:
                tl, ta, test_bacc, tc = test(model, criterion, test_loader, bAcc=True)
            else:
                tl, ta, tc = test(model, criterion, test_loader)

            result['test_loss'] = tl
            result['test_acc'] = ta
            if args.bAcc:
                result['test_bacc'] = test_bacc

            if step == 0:
                utils.print_row([k for k in result.keys()], colwidth=20)
            utils.print_row([v for v in result.values()], colwidth=20)

            with open(os.path.join(args.save_path, 'train_log.json'), 'a') as f:
                f.write(json.dumps(result, sort_keys=True) + "\n")

            # Track best
            if test_bacc > best_bacc:
                best_acc = ta
                best_bacc = test_bacc
                torch.save(model.state_dict(), os.path.join(args.save_path, 'test_best_ckpt.pth'))

    # Save last checkpoint
    torch.save(model.state_dict(), os.path.join(args.save_path, 'last_ckpt.pth'))

    if args.dataset == 'visda':
        # For VisDA, best_bacc is per-class accuracy
        return best_acc, selected_acc, result['test_acc'], best_bacc, best_bacc, test_bacc
    return best_acc, selected_acc, result['test_acc'], best_bacc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args):
    args.work_dir = os.path.join(f'./logs/{args.dataset}', f'{args.work_dir}/source{args.source}')
    os.makedirs(args.work_dir, exist_ok=True)

    with open(os.path.join(args.work_dir, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    utils.set_seed(args.seed)

    args.ckpts = get_ckpt(args)
    print(f"{args.dataset} source {args.source} -> target {args.target}")

    ret = train_on_target(args)
    if args.dataset == 'visda':
        best_acc, selected_acc, last_acc, best_pca, selected_pca, last_pca = ret
        print(f"Best Acc: {best_acc:.2f}  Per-class: {best_pca:.2f}")
    else:
        best_acc, selected_acc, last_acc, best_bacc = ret
        print(f"Best Acc: {best_acc:.2f}  bAcc: {best_bacc:.2f}")

    # Save results
    results = {
        f'source{args.source}@target{args.target}': {
            'best_acc': best_acc,
            'last_acc': last_acc,
            'best_bacc': best_bacc if args.dataset != 'visda' else best_pca,
        }
    }
    with open(os.path.join(args.work_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.work_dir}/results.json")


if __name__ == '__main__':
    args = parse_args()
    main(args)
