"""
Train a noised image classifier on ImageNet.
"""

import argparse
import os
import yaml

import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from sklearn.metrics import confusion_matrix
from pathlib import Path

from cg_diffusion import dist_util, logger
import cg_diffusion.filename_utils as filename_utils
from cg_diffusion.fp16_util import MixedPrecisionTrainer
from cg_diffusion.image_datasets import load_data_with_metadata
from cg_diffusion.resample import create_named_schedule_sampler
from cg_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    create_classifier_and_diffusion,
)
from cg_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict,split_microbatches

def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    if os.environ.get("AUTO_RESUME","False")=="True":
        log_dir=logger.get_dir()
        if not os.path.exists(log_dir):
            return None
        files=sorted(str(fp) for fp in Path(log_dir).glob("model*.pt"))
        if len(files)==0:
            return None
        latest_ckpt=files[-1]
        logger.log("automatically resuming from checkpoint",latest_ckpt)
        return latest_ckpt
    else:
        return None
    
def parse_class_mapping(class_mapping_str):
    if class_mapping_str=="default":
        return None
    mapping_dict=dict()
    for mapping in class_mapping_str.split(','):
        cl,cl_int=mapping.split(':')
        mapping_dict[cl]=int(cl_int)
    return mapping_dict

def parse_class_subset(class_subset_str):
    if class_subset_str=="default":
        return None
    return class_subset_str.split(',')

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()
    with open(os.path.join(logger.get_current().dir,"args.yaml"), "w") as f:
        yaml.dump(args.__dict__, f)
    
    logger.log("preparing files...")
    fp_list=[str(fp) for fp in Path(args.data_dir).glob("*.png")]
    metadata = dict()
    for fp in fp_list:
        metadata[fp] = filename_utils.parse_patch_filename(os.path.basename(fp))["slide_class"]
    
    args.class_subset = parse_class_subset(args.class_subset)
    if args.class_subset is not None:
        logger.log(f"using class subset {args.class_subset}")
    else:
        logger.log(f"using all classes in the directory")
    
    def class_subset(fp_list,metadata,class_subset):
        if class_subset is None:
            return fp_list,metadata
        else:
            fp_list = [fp for fp in fp_list if metadata[fp] in class_subset]
            metadata = {fp:metadata[fp] for fp in fp_list}
            return fp_list,metadata

    fp_list,metadata = class_subset(fp_list,metadata,args.class_subset)

    if args.class_mapping is None:
        from cg_diffusion.image_datasets import infer_class_mapping
        args.class_mapping = infer_class_mapping(metadata.values())
        args.num_classes = len(args.class_mapping)
    else:
        args.class_mapping = parse_class_mapping(args.class_mapping)
        assert len(args.class_mapping)>0, "invalid class mapping"
        if args.num_classes is None:
            args.num_classes = max(args.class_mapping.values())+1
        assert len(args.class_mapping)<=args.num_classes and min(args.class_mapping.values())>=0 and max(args.class_mapping.values())<args.num_classes, "invalid class mapping"
    
    if args.num_classes <= 30:
        logger.log(f"num_classes: {args.num_classes}, class mapping: {args.class_mapping}")
    else:
        logger.log(f"num_classes: {args.num_classes}")

    logger.log("creating classifier and diffusion...")
    model, diffusion = create_classifier_and_diffusion(
        **args_to_dict(args, classifier_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    if args.noised:
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion
        )

    resume_step = 0
    resume_checkpoint = find_resume_checkpoint() or args.resume_checkpoint
    if resume_checkpoint:
        resume_step = parse_resume_step_from_filename(resume_checkpoint)
        # if dist.get_rank() == 0:
        logger.log(
            f"loading model from checkpoint: {resume_checkpoint}... at {resume_step} step"
        )
        model.load_state_dict(
            dist_util.load_state_dict(
                resume_checkpoint, map_location=dist_util.dev()
            )
        )

    # Needed for creating correct EMAs and fp16 parameters.
    dist_util.sync_params(model.parameters())

    mp_trainer = MixedPrecisionTrainer(
        model=model, use_fp16=args.classifier_use_fp16, initial_lg_loss_scale=16.0
    )

    model = DDP(
        model,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )

    logger.log("creating data loader...")

    data = load_data_with_metadata(
        data_dir = None,
        fp_list = fp_list,
        batch_size = args.batch_size,
        image_size = args.image_size,
        metadata = metadata,
        clstr_to_int= lambda x: args.class_mapping[x],
        random_crop = True,
        random_flip = True,
        distributed = True,
        infinite = True,
        balance = True
    )

    if args.val_data_dir:
        val_fp_list = [str(fp) for fp in Path(args.val_data_dir).glob("*.png")]
        val_metadata = dict()
        for fp in val_fp_list:
            val_metadata[fp] = filename_utils.parse_patch_filename(os.path.basename(fp.name))["slide_class"]
            
        val_data = load_data_with_metadata(
            data_dir=None,
            fp_list=val_fp_list,
            batch_size=args.batch_size,
            image_size=args.image_size,
            metadata=val_metadata,
            random_crop=True,
            balance=True
        )
    else:
        val_data = None

    logger.log(f"creating optimizer...")
    opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)
    if resume_checkpoint:
        opt_checkpoint = bf.join(
            bf.dirname(resume_checkpoint), f"opt{resume_step:06}.pt"
        )
        logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
        opt.load_state_dict(
            dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
        )

    logger.log("training classifier model...")

    def forward_backward_log(data_loader, prefix="train"):
        batch, extra = next(data_loader)
        labels = extra["y"].to(dist_util.dev())

        batch = batch.to(dist_util.dev())
        # Noisy images
        if args.noised:
            t, _ = schedule_sampler.sample(batch.shape[0], dist_util.dev())
            batch = diffusion.q_sample(batch, t)
        else:
            t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())

        for i, (sub_batch, sub_labels, sub_t) in enumerate(
            split_microbatches(args.microbatch, batch, labels, t)
        ):
            logits = model(sub_batch, timesteps=sub_t)
            loss = F.cross_entropy(logits, sub_labels, reduction="none")

            losses = {}
            losses[f"{prefix}_loss"] = loss.detach()
            losses[f"{prefix}_acc@1"] = compute_top_k(
                logits, sub_labels, k=1, reduction="none"
            )
            if args.num_classes>5:
                losses[f"{prefix}_acc@5"] = compute_top_k(
                    logits, sub_labels, k=5, reduction="none"
                )
            pred = logits.argmax(dim=1).detach().cpu().numpy()
            gt = sub_labels.detach().cpu().numpy()
            cm = confusion_matrix(gt, pred, labels=range(args.num_classes))
            logger.log(cm)
            log_loss_dict(diffusion, sub_t, losses)
            del losses
            loss = loss.mean()
            if loss.requires_grad:
                if i == 0:
                    mp_trainer.zero_grad()
                mp_trainer.backward(loss * len(sub_batch) / len(batch))

    for step in range(args.iterations - resume_step):
        logger.logkv("step", step + resume_step)
        logger.logkv(
            "samples",
            (step + resume_step + 1) * args.batch_size * dist.get_world_size(),
        )
        if args.anneal_lr:
            set_annealed_lr(opt, args.lr, (step + resume_step) / args.iterations)
        forward_backward_log(data)
        mp_trainer.optimize(opt)
        if val_data is not None and not step % args.eval_interval:
            with th.no_grad():
                with model.no_sync():
                    model.eval()
                    forward_backward_log(val_data, prefix="val")
                    model.train()
        if not step % args.log_interval:
            logger.dumpkvs()
        if (
            dist.get_rank() == 0
            and not (step + resume_step) % args.save_interval
        ):
            logger.log("saving model...")
            save_model(mp_trainer, opt, step + resume_step)

    if dist.get_rank() == 0:
        logger.log("saving model...")
        save_model(mp_trainer, opt, step + resume_step)
    dist.barrier()


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(mp_trainer, opt, step):
    if dist.get_rank() == 0:
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join(logger.get_dir(), f"model{step:06d}.pt"),
        )
        th.save(opt.state_dict(), os.path.join(logger.get_dir(), f"opt{step:06d}.pt"))


def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)



    
def create_argparser():
    defaults = dict(
        data_dir="",
        val_data_dir="",
        class_subset = "default",
        class_mapping = None,
        noised=True,
        iterations=150000,
        lr=3e-4,
        weight_decay=0.0,
        anneal_lr=False,
        batch_size=4,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=10,
        eval_interval=5,
        save_interval=1000,
    )
    d = classifier_and_diffusion_defaults()
    del d["num_classes"]
    defaults.update(d)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument("--num_classes", type=int, default=None)
    return parser


if __name__ == "__main__":
    main()
