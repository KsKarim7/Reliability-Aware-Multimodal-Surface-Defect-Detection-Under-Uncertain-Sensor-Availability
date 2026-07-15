from builtins import len
import argparse

import torch.optim.lr_scheduler

from datasets import *
from datasets import dataset_classes
from utils.csv_utils import *
from utils.metrics import *
from utils.training_utils import *
from MISDD_MM import *
from utils.eval_utils import *
from utils.syn_anomaly import corrupt_batch, patch_labels, syn_patch_loss
from torchvision import transforms
import random
import time
import itertools
import wandb
from tqdm import tqdm

TASK = 'CLS'

def save_check_point(model, path):
    # persist feature galleries, text features and every learned module,
    # so results are reproducible from the checkpoint without retraining
    selected_keys = [
        'img_feature_gallery1',
        'img_feature_gallery2',
        'depth_feature_gallery1',
        'depth_feature_gallery2',
        'img_text_features',
        'depth_text_features',
    ]
    learned_prefixes = ('img_prompt_learner.', 'depth_prompt_learner.',
                        'missing_prompt_learner.', 'granular_text_guidance.')
    state_dict = model.state_dict()
    selected_state_dict = {k: v for k, v in state_dict.items()
                           if k in selected_keys or k.startswith(learned_prefixes)}

    torch.save(selected_state_dict, path)
    print(f"Model saved to {path}")


def _compute_grad_diagnostics(model, grad_innov1_buf, grad_innov2_buf, epoch, csv_path):
    """Log per-innovation gradient diagnostics once per epoch.
    Measures cosine similarity between Innovation 1 (CorrelatedPromptMLP)
    and Innovation 2 (DynamicPromptGenerator) gradient contributions.
    Only activates when full_model is running (all 4 innovations active).
    """
    import csv as _csv_mod, os
    pml = model.missing_prompt_learner
    total_grad = pml.image_prompt_complete.grad
    if total_grad is None:
        return
    total_norm = total_grad.norm().item()
    # Innovation 1 private param norms (CorrelatedPromptMLP layers)
    innov1_norms = [p.grad.norm().item() for m in pml.correlated_prompt_image
                    for p in m.parameters() if p.grad is not None]
    innov1_norm = float(sum(innov1_norms) / len(innov1_norms)) if innov1_norms else 0.0
    # Innovation 2 private param norms (DynamicPromptGenerator)
    innov2_norms = [p.grad.norm().item() for p in pml.dynamic_image_gen.parameters()
                    if p.grad is not None]
    innov2_norm = float(sum(innov2_norms) / len(innov2_norms)) if innov2_norms else 0.0
    # Cosine similarity between hooked gradient buffers
    cos_sim = float("nan")
    if grad_innov1_buf[0] is not None and grad_innov2_buf[0] is not None:
        g1 = grad_innov1_buf[0].flatten().float()
        g2 = grad_innov2_buf[0].flatten().float()
        denom = g1.norm() * g2.norm()
        if denom > 1e-8:
            cos_sim = (g1 @ g2 / denom).item()
    # Write to CSV
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = _csv_mod.writer(f)
        if write_header:
            writer.writerow(["epoch", "cos_sim_innov1_innov2",
                             "innov1_grad_norm", "innov2_grad_norm",
                             "total_prompt_grad_norm"])
        writer.writerow([epoch, f"{cos_sim:.6f}", f"{innov1_norm:.6f}",
                         f"{innov2_norm:.6f}", f"{total_norm:.6f}"])

def fit(model,
        args,
        dataloader: DataLoader,
        device: str,
        check_path: str,
        train_data: DataLoader,
        ):

    # change the model into eval mode
    model.eval_mode()

    # the visual (missing-aware) prompts get their own lr: at the full text-side lr,
    # joint normal-only training collapses the text-anchor discrimination
    _vp_lr = args.visual_prompt_lr if args.visual_prompt_lr is not None else args.lr
    text_params = itertools.chain(model.img_prompt_learner.parameters(), model.depth_prompt_learner.parameters(), model.granular_text_guidance.parameters() if hasattr(model, 'granular_text_guidance') else [])

    optimizer = torch.optim.SGD([
        {'params': list(text_params), 'lr': args.lr},
        {'params': list(model.missing_prompt_learner.parameters()), 'lr': _vp_lr},
    ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.Epoch, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss().to(device)
    criterion_tip = TripletLoss(margin=0.0)

    n_train = len(train_data.dataset)
    for epoch in range(args.Epoch):
        # TRAIN — one optimizer step per epoch, accumulated over microbatches.
        # This preserves the 50-step full-batch optimization budget the protocol
        # was tuned for; per-microbatch stepping (~9x more steps at the same lr)
        # overtrains the text anchors until normal/abnormal discrimination inverts.
        optimizer.zero_grad()
        for (img, pc, depth, mask, label, name, img_type, missing_flag) in tqdm(train_data, ncols=100, desc=f'{args.dataset}/{args.class_name}, missing {args.missing_type}-{args.missing_rate}, Epoch {epoch}/{args.Epoch}, training'):
            img = [model.img_transform(Image.fromarray(cv2.cvtColor(f.numpy(), cv2.COLOR_BGR2RGB))) for f in img]
            # pc = [p for p in pc]
            depth = [d for d in depth]
            # print(type(img), len(img))
            # print(type(pc), len(pc))
            # print(type(depth), len(depth))
            img = torch.stack(img, dim=0).to(device)
            # pc = torch.stack(pc, dim=0).to(device)
            depth = torch.stack(depth, dim=0).to(device)

            # img = img[0:1, :, :, :].to(device)
            img = img.to(device)
            # pc = pc.to(device)
            depth = depth.to(device)

            # normal_text_prompt, abnormal_text_prompt_manual, abnormal_text_prompt_learned = model.prompt_learner()
            img_normal_text_prompt, img_abnormal_text_prompt_manual, img_abnormal_text_prompt_learned = model.img_prompt_learner()
            depth_normal_text_prompt, depth_abnormal_text_prompt_manual, depth_abnormal_text_prompt_learned = model.depth_prompt_learner()
            all_prompts_image, all_prompts_depth = model.missing_prompt_learner(missing_flag, raw_image=img, raw_depth=depth)

            img_feature, _, _, _ = model.encode_image_missing(img, all_prompts_image, missing_flag)
            img_normal_text_features = model.encode_text_embedding(img_normal_text_prompt, model.img_tokenized_normal_prompts)
            img_abnormal_text_features_manual = model.encode_text_embedding(img_abnormal_text_prompt_manual, model.img_tokenized_abnormal_prompts_manual)
            img_abnormal_text_features_learned = model.encode_text_embedding(img_abnormal_text_prompt_learned, model.img_tokenized_abnormal_prompts_learned)
            img_abnormal_text_features = torch.cat([img_abnormal_text_features_manual, img_abnormal_text_features_learned], dim=0)
            img_mean_ad_manual = torch.mean(F.normalize(img_abnormal_text_features_manual, dim=-1), dim=0)
            img_mean_ad_learned = torch.mean(F.normalize(img_abnormal_text_features_learned, dim=-1), dim=0)
            img_loss_match_abnormal = (img_mean_ad_manual - img_mean_ad_learned).norm(dim=0) ** 2.0
            img_normal_text_features_ahchor = img_normal_text_features.mean(dim=0).unsqueeze(0)
            img_normal_text_features_ahchor = img_normal_text_features_ahchor / img_normal_text_features_ahchor.norm(dim=-1, keepdim=True)
            img_abnormal_text_features_ahchor = img_abnormal_text_features.mean(dim=0).unsqueeze(0)
            img_abnormal_text_features_ahchor = img_abnormal_text_features_ahchor / img_abnormal_text_features_ahchor.norm(dim=-1, keepdim=True)
            img_abnormal_text_features = img_abnormal_text_features / img_abnormal_text_features.norm(dim=-1, keepdim=True)
            img_l_pos = torch.einsum('nc,cm->nm', img_feature, img_normal_text_features_ahchor.transpose(0, 1))
            img_l_neg_v2t = torch.einsum('nc,cm->nm', img_feature, img_abnormal_text_features.transpose(0, 1))
            img_logits_v2t = torch.cat([img_l_pos, img_l_neg_v2t], dim=-1) * model.model.logit_scale
            img_target_v2t = torch.zeros([img_logits_v2t.shape[0]], dtype=torch.long).to(device)
            img_loss_v2t = criterion(img_logits_v2t, img_target_v2t)
            img_trip_loss = criterion_tip(img_feature, img_normal_text_features_ahchor, img_abnormal_text_features_ahchor)
            img_loss = img_loss_v2t + img_trip_loss + img_loss_match_abnormal * args.lambda1

            depth_feature, _, _, _ = model.encode_image_missing(depth, all_prompts_depth, missing_flag)
            depth_normal_text_features = model.encode_text_embedding(depth_normal_text_prompt, model.depth_tokenized_normal_prompts)
            depth_abnormal_text_features_manual = model.encode_text_embedding(depth_abnormal_text_prompt_manual, model.depth_tokenized_abnormal_prompts_manual)
            depth_abnormal_text_features_learned = model.encode_text_embedding(depth_abnormal_text_prompt_learned, model.depth_tokenized_abnormal_prompts_learned)
            depth_abnormal_text_features = torch.cat([depth_abnormal_text_features_manual, depth_abnormal_text_features_learned], dim=0)
            depth_mean_ad_manual = torch.mean(F.normalize(depth_abnormal_text_features_manual, dim=-1), dim=0)
            depth_mean_ad_learned = torch.mean(F.normalize(depth_abnormal_text_features_learned, dim=-1), dim=0)
            depth_loss_match_abnormal = (depth_mean_ad_manual - depth_mean_ad_learned).norm(dim=0) ** 2.0
            depth_normal_text_features_ahchor = depth_normal_text_features.mean(dim=0).unsqueeze(0)
            depth_normal_text_features_ahchor = depth_normal_text_features_ahchor / depth_normal_text_features_ahchor.norm(dim=-1, keepdim=True)
            depth_abnormal_text_features_ahchor = depth_abnormal_text_features.mean(dim=0).unsqueeze(0)
            depth_abnormal_text_features_ahchor = depth_abnormal_text_features_ahchor / depth_abnormal_text_features_ahchor.norm(dim=-1, keepdim=True)
            depth_abnormal_text_features = depth_abnormal_text_features / depth_abnormal_text_features.norm(dim=-1, keepdim=True)
            depth_l_pos = torch.einsum('nc,cm->nm', depth_feature, depth_normal_text_features_ahchor.transpose(0, 1))
            depth_l_neg_v2t = torch.einsum('nc,cm->nm', depth_feature, depth_abnormal_text_features.transpose(0, 1))
            depth_logits_v2t = torch.cat([depth_l_pos, depth_l_neg_v2t], dim=-1) * model.model.logit_scale
            depth_target_v2t = torch.zeros([depth_logits_v2t.shape[0]], dtype=torch.long).to(device)
            depth_loss_v2t = criterion(depth_logits_v2t, depth_target_v2t)
            depth_trip_loss = criterion_tip(depth_feature, depth_normal_text_features_ahchor, depth_abnormal_text_features_ahchor)
            depth_loss = depth_loss_v2t + depth_trip_loss + depth_loss_match_abnormal * args.lambda1
            
            if hasattr(model, 'missing_prompt_learner') and hasattr(model, 'granular_text_guidance'):
                granular_loss = model.granular_text_guidance.compute_alignment_loss(
                    all_prompts_image, all_prompts_depth)
                # Linear warmup: granular loss weight grows from 0 to gran_weight over training
                # This lets the main SCL loss stabilize direction before auxiliary loss
                # adds gradient pressure, reducing seed-to-seed interference
                gran_weight = args.gran_weight * (epoch + 1) / args.Epoch
                loss = args.img_lambda * img_loss + args.depth_lambda * depth_loss + gran_weight * granular_loss
            else:
                loss = args.img_lambda * img_loss + args.depth_lambda * depth_loss
                granular_loss = None
                gran_weight = 0.0

            # synthetic-anomaly contrastive term (Gate 3): corrupted twins give the
            # patch-textual pathway its first real anomaly signal
            syn_loss = None
            if args.syn_anomaly:
                sel = torch.rand(img.shape[0]) < 0.5
                if sel.any():
                    s_img, s_dep, s_masks = corrupt_batch(img[sel], depth[sel])
                    s_flags = missing_flag[sel]
                    s_api, s_apd = model.missing_prompt_learner(s_flags, raw_image=s_img, raw_depth=s_dep)
                    _, s_maps_i, _, _ = model.encode_image_missing(s_img, s_api, s_flags)
                    _, s_maps_d, _, _ = model.encode_image_missing(s_dep, s_apd, s_flags)
                    s_labels = patch_labels(s_masks, model.grid_size)
                    syn_loss = (syn_patch_loss(s_maps_i, s_labels, img_normal_text_features_ahchor,
                                               img_abnormal_text_features_ahchor, model.model.logit_scale)
                                + syn_patch_loss(s_maps_d, s_labels, depth_normal_text_features_ahchor,
                                                 depth_abnormal_text_features_ahchor, model.model.logit_scale))
                    loss = loss + args.syn_weight * syn_loss

            wandb.log({
                'loss': loss.item(), 
                'img_loss_v2t': img_loss_v2t.item(),
                'img_trip_loss': img_trip_loss.item(), 
                'depth_loss_v2t': depth_loss_v2t.item(), 
                'depth_trip_loss': depth_trip_loss.item(), 
                'img_loss_match_abnormal': img_loss_match_abnormal.item(),
                'depth_loss_match_abnormal': depth_loss_match_abnormal.item(),
                'granular_loss': granular_loss.item() if granular_loss is not None else 0.0,
                'gran_weight': gran_weight,
                'syn_loss': syn_loss.item() if syn_loss is not None else 0.0,
            })

            # accumulate a full-batch-equivalent gradient: weight each microbatch
            # by its share of the training set
            (loss * (img.shape[0] / n_train)).backward()

        # clip strength is set explicitly per run via --max_norm (no hidden config detection)
        _clip_norm = args.max_norm
        torch.nn.utils.clip_grad_norm_(model.missing_prompt_learner.parameters(), max_norm=_clip_norm)
        torch.nn.utils.clip_grad_norm_(model.img_prompt_learner.parameters(), max_norm=_clip_norm)
        torch.nn.utils.clip_grad_norm_(model.depth_prompt_learner.parameters(), max_norm=_clip_norm)
        if hasattr(model, 'granular_text_guidance'):
            torch.nn.utils.clip_grad_norm_(model.granular_text_guidance.parameters(), max_norm=_clip_norm)
        optimizer.step()
        # Log gradient diagnostics once per epoch (grads from this epoch's step are still present)
        if args.grad_diag:
            _diag_csv = check_path.replace('.pt', f'_grad_diag_seed{args.seed}.csv')
            _compute_grad_diagnostics(model, [None], [None], epoch, _diag_csv)
        scheduler.step()

    # report learned residual scales (gamma starts at 1e-2; where it ends up is a result)
    try:
        pml = model.missing_prompt_learner
        if hasattr(pml.dynamic_image_gen, 'gamma'):
            corr_g = [round(m.gamma.item(), 4) for m in pml.correlated_prompt_image]
            print(f'[gamma-diag] corr_image={corr_g} '
                  f'dyn_image={pml.dynamic_image_gen.gamma.item():.4f} '
                  f'dyn_depth={pml.dynamic_depth_gen.gamma.item():.4f}')
    except Exception:
        pass

    # training finished: build the feature galleries with the TRAINED prompts so the
    # gallery and test-time representations match (test features are prompted too)
    img_features1, img_features2 = [], []
    depth_features1, depth_features2 = [], []
    with torch.no_grad():
        for (img, pc, depth, mask, label, name, img_type, missing_flag) in tqdm(train_data, ncols=100, desc=f'{args.dataset}/{args.class_name}, missing {args.missing_type}-{args.missing_rate}, building feature gallery'):
            img = [model.img_transform(Image.fromarray(cv2.cvtColor(f.numpy(), cv2.COLOR_BGR2RGB))) for f in img]
            depth = [d for d in depth]
            img = torch.stack(img, dim=0).to(device)
            depth = torch.stack(depth, dim=0).to(device)
            all_prompts_image, all_prompts_depth = model.missing_prompt_learner(missing_flag, raw_image=img, raw_depth=depth)
            _, _, img_feature_map1, img_feature_map2 = model.encode_image_missing(img, all_prompts_image, missing_flag)
            _, _, depth_feature_map1, depth_feature_map2 = model.encode_image_missing(depth, all_prompts_depth, missing_flag)
            img_features1.append(img_feature_map1)
            img_features2.append(img_feature_map2)
            depth_features1.append(depth_feature_map1)
            depth_features2.append(depth_feature_map2)
    model.build_image_feature_gallery(torch.cat(img_features1, dim=0), torch.cat(img_features2, dim=0))
    model.build_depth_feature_gallery(torch.cat(depth_features1, dim=0), torch.cat(depth_features2, dim=0))

    # build text feature galleries once, then evaluate the final model.
    # Evaluation happens once at the final epoch only — no best-epoch selection on the
    # test set — so the reported number is an unbiased estimate of the converged model.
    model.build_img_text_feature_gallery()
    model.build_depth_text_feature_gallery()

    # did the synthetic objective actually learn the synthetic task? (train-side
    # check, no test-set contact; distinguishes "didn't learn" from "didn't transfer")
    if args.syn_anomaly:
        with torch.no_grad():
            (img, pc, depth, mask, label, name, img_type, missing_flag) = next(iter(train_data))
            img = torch.stack([model.img_transform(Image.fromarray(cv2.cvtColor(f.numpy(), cv2.COLOR_BGR2RGB))) for f in img], 0).to(device)
            depth = torch.stack([d for d in depth], 0).to(device)
            s_img, s_dep, s_masks = corrupt_batch(img, depth)
            s_api, s_apd = model.missing_prompt_learner(missing_flag, raw_image=s_img, raw_depth=s_dep)
            _, s_maps_i, _, _ = model.encode_image_missing(s_img, s_api, missing_flag)
            s_labels = patch_labels(s_masks, model.grid_size)
            tf = s_maps_i / s_maps_i.norm(dim=-1, keepdim=True)
            probs = (model.model.logit_scale * tf @ model.img_text_features.t()).softmax(dim=-1)[..., 1]
            try:
                _auc = roc_auc_score(s_labels.cpu().numpy().reshape(-1), probs.cpu().numpy().reshape(-1))
                print(f'[syn-diag] corrupted-vs-clean patch AUROC (train-side): {_auc*100:.2f}')
            except ValueError:
                print('[syn-diag] degenerate labels in diag batch')

    # TEST (final model)
    scores_img = []
    score_maps = []
    test_imgs = []
    test_depths = []
    gt_list = []
    gt_mask_list = []
    names = []
    for (img, pc, depth, mask, label, name, img_type, missing_flag) in tqdm(dataloader, ncols=100, desc=f'{args.dataset}/{args.class_name}, missing {args.missing_type}-{args.missing_rate}, final testing'):

        # same BGR->RGB conversion as training (test images previously stayed BGR)
        img = [model.img_transform(Image.fromarray(cv2.cvtColor(f.numpy(), cv2.COLOR_BGR2RGB))) for f in img]
        depth = [d for d in depth]
        img = torch.stack(img, dim=0)
        depth = torch.stack(depth, dim=0)
        with torch.no_grad():
            all_prompts_image, all_prompts_depth = model.missing_prompt_learner(missing_flag, raw_image=img, raw_depth=depth)

        for d, t, n, l, m in zip(img, depth, name, label, mask):
            test_imgs += [denormalization(d.cpu().numpy())]
            test_depths += [denormalization_depth(t.cpu().numpy())]
            l = l.numpy()
            m = m.numpy()
            m[m > 0] = 1

            names += [n]
            gt_list += [l]
            gt_mask_list += [m]

        img = img.to(device)
        depth = depth.to(device)
        score_img, score_map = model(args, img, depth, 'cls', all_prompts_image, all_prompts_depth, missing_flag)
        score_maps += score_map
        scores_img += score_img

    test_imgs, test_depths, score_maps, gt_mask_list = specify_resolution(test_imgs, test_depths, score_maps, gt_mask_list, resolution=(args.resolution, args.resolution))
    # component diagnostics: the fused score hides which branch is broken
    try:
        _y = np.asarray(gt_list, dtype=int)
        _text_scores = np.asarray(scores_img)
        _map_scores = np.asarray(score_maps).reshape(len(score_maps), -1).max(axis=1)
        print(f'[component-diag] textual-only AUROC: {roc_auc_score(_y, _text_scores)*100:.2f} | map-only AUROC: {roc_auc_score(_y, _map_scores)*100:.2f}')
    except Exception:
        pass
    result_dict = metric_cal_img(np.array(scores_img), gt_list, np.array(score_maps),
                                 score_mode=args.img_score_mode)
    try:
        pix_result_dict = metric_cal_pix(np.array(score_maps), gt_mask_list)
        result_dict.update(pix_result_dict)
    except Exception:
        # keep the zero placeholders so the CSV shape is stable, but fail LOUDLY:
        # a zero p_roc/pro_auc row must never pass silently again
        import traceback, sys
        print(f'!!! PIXEL METRIC FAILURE for {args.class_name} (p_roc/pro_auc set to 0.0):', file=sys.stderr)
        traceback.print_exc()
        result_dict['p_roc'] = 0.0
        result_dict['pro_auc'] = 0.0

    print(f'===========================Image-AUROC: {result_dict["i_roc"]:.2f} | P-AUROC: {result_dict.get("p_roc", 0):.2f} | AUPRO: {result_dict.get("pro_auc", 0):.2f}')
    save_check_point(model, check_path)

    wandb.log({
        'Image-AUROC': result_dict['i_roc'],
    })

    return result_dict


def main(args):
    kwargs = vars(args)

    if kwargs['seed'] is None:
        kwargs['seed'] = 111

    setup_seed(kwargs['seed'])

    if kwargs['use_cpu'] == 0:
        device = f"cuda:0"
    else:
        device = f"cpu"
    kwargs['device'] = device

    wandb.init(
        project = 'Prompt-RGB_DEPTH-CLS-Missing-V3',
        name = f'{args.dataset}-{args.class_name}-{args.missing_type}-{args.missing_rate}-{args.seed}-{args.img_lambda}-{args.pc_lambda}-{time.time()}',
    )

    # prepare the experiment dir
    _, csv_path, check_path = get_dir_from_args(TASK, **kwargs)

    # get the train dataloader
    train_dataloader, train_dataset_inst, train_dataset_len = get_dataloader_from_args(phase='train', perturbed=False, **kwargs)

    # get the test dataloader
    test_dataloader, test_dataset_inst, test_dataset_len = get_dataloader_from_args(phase='test', perturbed=False, **kwargs)

    kwargs['out_size_h'] = kwargs['resolution']
    kwargs['out_size_w'] = kwargs['resolution']
    kwargs['size'] = train_dataset_len

    # get the model
    model = MISDD_MM(**kwargs)
    model = model.to(device)

    # as the pro metric calculation is costly, we only calculate it in the last evaluation
    metrics = fit(model, args, test_dataloader, device, check_path=check_path, train_data=train_dataloader)

    i_roc = round(metrics['i_roc'], 2)
    object = kwargs['class_name']
    print(f'Object:{object} =========================== Image-AUROC:{i_roc}\n')

    save_metric(metrics, dataset_classes[kwargs['dataset']], kwargs['class_name'],
                kwargs['dataset'], csv_path)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_args():
    parser = argparse.ArgumentParser(description='Anomaly detection')
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa', 'mvtec3d', 'eyescandies'])
    parser.add_argument('--class_name', type=str, default='carpet')

    parser.add_argument('--img-resize', type=int, default=240)
    parser.add_argument('--img-cropsize', type=int, default=240)
    parser.add_argument('--resolution', type=int, default=400)

    # gradients now flow through the encoder, so full-dataset batches no longer fit in VRAM
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--vis', type=str2bool, choices=[True, False], default=False)
    parser.add_argument("--root-dir", type=str, default="./result")
    parser.add_argument("--load-memory", type=str2bool, default=True)
    parser.add_argument("--cal-pro", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--gpu-id", type=int, default=0)

    # pure test
    parser.add_argument("--pure-test", type=str2bool, default=False)

    # method related parameters
    parser.add_argument('--k-shot', type=int, default=1)
    parser.add_argument('--missing_type', type=str, default='both')
    parser.add_argument('--missing_rate', type=float, default=0.3)
    parser.add_argument("--backbone", type=str, default="ViT-B-16-plus-240",
                        choices=['ViT-B-16-plus-240', 'ViT-B-16'])
    parser.add_argument("--pretrained_dataset", type=str, default="laion400m_e32")

    parser.add_argument("--use-cpu", type=int, default=0)

    # prompt tuning hyper-parameter
    parser.add_argument("--n_ctx", type=int, default=4)
    parser.add_argument("--n_ctx_ab", type=int, default=1)
    parser.add_argument("--n_pro", type=int, default=3)
    parser.add_argument("--n_pro_ab", type=int, default=4)
    parser.add_argument("--Epoch", type=int, default=50)
    parser.add_argument("--img_lambda", type=float, default=0.5)
    parser.add_argument("--pc_lambda", type=float, default=0.5)
    parser.add_argument("--depth_lambda", type=float, default=0.5)
    parser.add_argument("--missing_prompt_length", type=int, default=36)
    parser.add_argument("--missing_prompt_depth", type=int, default=6)

    # optimizer
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--visual_prompt_lr", type=float, default=None,
                        help="Separate lr for the missing-aware visual prompt learner (default: same as --lr)")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)

    # loss hyper parameter
    parser.add_argument("--lambda1", type=float, default=0.001)
    parser.add_argument("--gran_weight", type=float, default=0.1,
                        help="Maximum weight for granular text guidance loss (linearly warmed up from 0)")
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--grad_diag", type=str2bool, default=False,
                        help="Log per-innovation gradient norm diagnostics once per epoch")
    parser.add_argument("--img_score_mode", type=str, default='harmonic', choices=['harmonic', 'map'],
                        help="Image-level score: 'harmonic' fuses map with the global textual score "
                             "(original protocol), 'map' uses the map branch alone")
    parser.add_argument("--map_knn", type=int, default=1,
                        help="k nearest gallery patches averaged per test patch in the map score "
                             "(k=3 selected on seed 111, frozen before seeds 222/333)")
    parser.add_argument("--syn_anomaly", type=str2bool, default=False,
                        help="Gate 3: add the synthetic-anomaly contrastive patch objective")
    parser.add_argument("--syn_weight", type=float, default=0.2,
                        help="weight of the synthetic-anomaly patch loss")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    import os

    args = get_args()
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu_id}"
    main(args)
