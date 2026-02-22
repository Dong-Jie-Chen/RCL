import utils
import torch

def train_one_step_fixmatch(model, criterion, optimizer, x_w, x_s, ulb_mask, y, args):
    model.train()

    # Labeled data
    x_w, x_s, y = x_w.cuda(), x_s.cuda(), y.cuda()
    output_labeled = model(x_w)
    loss_labeled = criterion(output_labeled, y)

    # Unlabeled data
    x_unlabeled_weak = x_w[ulb_mask]
    x_unlabeled_strong = x_s[ulb_mask]

    # Weak augmentation
    output_unlabeled_weak = model(x_unlabeled_weak)
    pseudo_labels = torch.argmax(output_unlabeled_weak, dim=1)

    # Strong augmentation
    output_unlabeled_strong = model(x_unlabeled_strong)

    # Pseudo-labeling
    loss_unlabeled = criterion(output_unlabeled_strong, pseudo_labels).mean()

    # Total loss
    loss = loss_labeled + args.lambda_u * loss_unlabeled

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    acc, _ = utils.accuracy(output_labeled, y, topk=(1, 5))
    return loss.item(), loss_unlabeled.item(), acc