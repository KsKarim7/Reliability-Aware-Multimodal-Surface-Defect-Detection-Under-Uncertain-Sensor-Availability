import torch
import torch.nn.functional as F


def _blob_mask(h, w, device, area_lo=0.02, area_hi=0.15):
    """smooth-noise blob mask covering a random fraction of the image"""
    scale = int(torch.randint(4, 9, (1,)).item())
    g = torch.randn(1, 1, scale, scale, device=device)
    m = F.interpolate(g, size=(h, w), mode='bicubic', align_corners=False)[0, 0]
    frac = area_lo + torch.rand(1).item() * (area_hi - area_lo)
    thresh = torch.quantile(m.flatten(), 1.0 - frac)
    return (m > thresh).float()


def _rect_mask(h, w, device, area_lo=0.02, area_hi=0.15):
    frac = area_lo + torch.rand(1).item() * (area_hi - area_lo)
    rh = max(4, int((frac ** 0.5) * h * (0.5 + torch.rand(1).item())))
    rw = max(4, int(frac * h * w / rh))
    rw = min(rw, w - 1)
    y = torch.randint(0, h - rh, (1,)).item()
    x = torch.randint(0, w - rw, (1,)).item()
    m = torch.zeros(h, w, device=device)
    m[y:y + rh, x:x + rw] = 1.0
    return m


def corrupt_batch(img, depth):
    """Return corrupted twins of (img, depth) plus per-pixel corruption masks.
    img/depth: [B, 3, H, W] transformed tensors. Same mask on both modalities;
    RGB gets transplanted content, depth gets the transplant plus a scalar offset.
    """
    B, _, H, W = img.shape
    syn_img = img.clone()
    syn_depth = depth.clone()
    masks = torch.zeros(B, H, W, device=img.device)
    for b in range(B):
        if torch.rand(1).item() < 0.5:
            m = _rect_mask(H, W, img.device)
        else:
            m = _blob_mask(H, W, img.device)
        # transplant source: shifted copy of the same image (CutPaste) or another
        # image in the batch (blob fill)
        src = img[(b + 1) % B] if B > 1 and torch.rand(1).item() < 0.5 else torch.roll(
            img[b], shifts=(H // 3, W // 3), dims=(1, 2))
        src_d = depth[(b + 1) % B] if B > 1 else torch.roll(depth[b], shifts=(H // 3, W // 3), dims=(1, 2))
        syn_img[b] = img[b] * (1 - m) + src * m
        d_off = (torch.rand(1, device=img.device).item() - 0.5) * 0.8 * depth[b].abs().max()
        syn_depth[b] = depth[b] * (1 - m) + (src_d + d_off) * m
        masks[b] = m
    return syn_img, syn_depth, masks


def patch_labels(masks, grid_size, thresh=0.3):
    """downsample pixel masks to the patch grid: a patch is anomalous if more
    than `thresh` of its pixels are corrupted. Returns [B, L] float labels."""
    B = masks.shape[0]
    m = F.adaptive_avg_pool2d(masks.unsqueeze(1), grid_size)
    return (m.reshape(B, -1) > thresh).float()


def syn_patch_loss(token_features, labels, normal_anchor, abnormal_anchor, logit_scale):
    """CE over the 2-anchor softmax for every patch; corrupted patches target the
    abnormal anchor, clean patches the normal one. token_features: [B, L, D]."""
    tf = token_features / token_features.norm(dim=-1, keepdim=True)
    anchors = torch.cat([normal_anchor, abnormal_anchor], dim=0)  # [2, D]
    logits = logit_scale * tf @ anchors.t()                       # [B, L, 2]
    return F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1).long())
