"""
Curriculum learning strategies for the three-stage RCL pipeline.

Stage 2 (SMKE): curriculum_2and3_self_correct
Stage 3 (MMR):  curriculum_teacher_class_filter_conf_th_ssl  (3 MLLMs)
                curriculum_teacher_class_filter_conf_th_ssl_4mllms (4 MLLMs)
"""

import torch


def curriculum_2and3_self_correct(model, x, y, args):
    """Stage 2 – SMKE: Self-correction curriculum.

    When the model is confident enough (above p_th), override the MLLM
    pseudo-label with the model's own prediction (self-correction).
    """
    x, y = x.cuda(), y.cuda()

    logits = model(x)
    output = torch.softmax(logits, dim=-1)
    pseudo_label = torch.argmax(logits, dim=-1)
    confidence_mask = output.max(1)[0].ge(args.p_th)

    mask_disagree = (pseudo_label != y)
    mask_disagree = mask_disagree.sum().item() / len(mask_disagree)

    # Self-correct: replace MLLM labels with model predictions where confident
    y_after_vote = y.clone()
    y_after_vote[confidence_mask] = pseudo_label[confidence_mask]

    return x, y_after_vote, mask_disagree


def curriculum_teacher_class_filter_conf_th_ssl(model, x, y_teacher1, y_teacher2, y_teacher3, args):
    """Stage 3 – MMR: Teacher-guided class filtering with confidence threshold (3 MLLMs).

    Restricts pseudo-labels to classes nominated by at least one MLLM teacher.
    If the model is confident enough (above p_th), the original pseudo-label
    is kept instead.  Returns an unlabeled mask for FixMatch SSL.
    """
    x = x.cuda()
    y_teacher1, y_teacher2, y_teacher3 = y_teacher1.cuda(), y_teacher2.cuda(), y_teacher3.cuda()
    logits = model(x)

    probs = torch.softmax(logits, dim=1)
    origin_pseudo_label = torch.argmax(probs, dim=-1)

    # Mask: keep only the classes assigned by each teacher
    mask = torch.zeros_like(probs)
    mask[torch.arange(x.size(0)), y_teacher1] = 1
    mask[torch.arange(x.size(0)), y_teacher2] = 1
    mask[torch.arange(x.size(0)), y_teacher3] = 1

    masked_probs = probs * mask
    pseudo_label = torch.argmax(masked_probs, dim=-1)

    # If model is confident, trust its own prediction
    confidence_mask = probs.max(1)[0].ge(args.p_th)
    pseudo_label[confidence_mask] = origin_pseudo_label[confidence_mask]

    label_changed_rate = (origin_pseudo_label != pseudo_label).sum().item() / len(origin_pseudo_label)

    ulb_mask = torch.ones_like(confidence_mask).to(torch.bool)
    return x, ulb_mask, pseudo_label, label_changed_rate


def curriculum_teacher_class_filter_conf_th_ssl_4mllms(model, x, y_teacher1, y_teacher2, y_teacher3, y_teacher4, args):
    """Stage 3 – MMR variant: Teacher-guided class filtering with 4 MLLMs.

    Same logic as the 3-MLLM version, but incorporates a fourth teacher.
    """
    x = x.cuda()
    y_teacher1, y_teacher2, y_teacher3, y_teacher4 = (
        y_teacher1.cuda(), y_teacher2.cuda(), y_teacher3.cuda(), y_teacher4.cuda()
    )
    logits = model(x)

    probs = torch.softmax(logits, dim=1)
    origin_pseudo_label = torch.argmax(probs, dim=-1)

    mask = torch.zeros_like(probs)
    mask[torch.arange(x.size(0)), y_teacher1] = 1
    mask[torch.arange(x.size(0)), y_teacher2] = 1
    mask[torch.arange(x.size(0)), y_teacher3] = 1
    mask[torch.arange(x.size(0)), y_teacher4] = 1

    masked_probs = probs * mask
    pseudo_label = torch.argmax(masked_probs, dim=-1)

    confidence_mask = probs.max(1)[0].ge(args.p_th)
    pseudo_label[confidence_mask] = origin_pseudo_label[confidence_mask]

    label_changed_rate = (origin_pseudo_label != pseudo_label).sum().item() / len(origin_pseudo_label)

    ulb_mask = torch.ones_like(confidence_mask).to(torch.bool)
    return x, ulb_mask, pseudo_label, label_changed_rate
