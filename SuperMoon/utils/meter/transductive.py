def trans_class_acc(pred, target, mask):
    mask = mask.bool()
    pred = pred[mask].max(1)[1]
    acc = pred.eq(target[mask]).sum().item() / mask.sum().item()
    return acc


def trans_iou_socre(pred, target, mask):
    mask = mask.bool()
    ious = []
    n_class = target.max().item() + 1
    pred = pred[mask].max(1)[1]
    target = target[mask]

    # IOU for background class ("0")
    for _c in range(1, n_class):
        pred_idx = pred == _c
        target_idx = target == _c
        intersection = (pred_idx & target_idx).sum().float().item()
        union = (pred_idx | target_idx).sum().float().item()
        ious.append((intersection + 1e-6) / (union + 1e-6))

    return ious
