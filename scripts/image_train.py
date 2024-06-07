"""
Train a diffusion model on images.
"""
import os
import sys
repo_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_root_dir)

import yaml
import argparse
from pathlib import Path

import torch as th
from cg_diffusion import dist_util, logger
from cg_diffusion.image_datasets import load_data_with_metadata
from cg_diffusion.resample import create_named_schedule_sampler
from cg_diffusion.script_util import (
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    args_to_dict,
    add_dict_to_argparser,
)
from cg_diffusion.train_util import TrainLoop
import cg_diffusion.filename_utils as filename_utils

import inspect

def test_classifier(classifier,image_size):
    bs = 10
    device = next(classifier.parameters()).device
    mock_inp_x = th.rand(bs,3,image_size,image_size).to(device)
    mock_inp_t = th.rand(bs).to(device)
    mock_out = classifier(mock_inp_x,mock_inp_t)
    return mock_out

def parse_args():
    args = create_argparser().parse_args()
    return args

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
    args = parse_args()
    logger.configure()
    # save args to file by yaml
    with open(os.path.join(logger.get_current().dir,"args.yaml"), "w") as f:
        yaml.dump(args.__dict__, f)

    dist_util.setup_dist()

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

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")

    data = load_data_with_metadata(
        data_dir=None,
        fp_list=fp_list,
        batch_size=args.batch_size,
        image_size=args.image_size,
        metadata=metadata,
        clstr_to_int= lambda x: args.class_mapping[x],
        random_crop = False,
        random_flip = True,
        distributed = True,
        infinite=True
    )

    def parse_cls_str(s,class_mapping):
        if s.isnumeric():
            cl = int(s)
            assert cl>=0 and cl<len(class_mapping)
            return int(s)
        else:
            return class_mapping[s]
        
    if args.produce_diffusion_samples:
        if args.specific_classes_to_sample is None:
            args.specific_classes_to_sample=range(args.num_classes)
        else:
            if len(args.specific_classes_to_sample)>0:
                args.specific_classes_to_sample=[parse_cls_str(cl,args.class_mapping) for cl in args.specific_classes_to_sample.split(',')]
            else:
                args.specific_classes_to_sample=None
        if args.specific_classes_to_sample is not None:
            logger.log(f"need to sample diffusion model during training. classes to sample {args.specific_classes_to_sample}. loading classifier...")
        else:
            logger.log(f"need to sample diffusion model during training. loading classifier...")
        classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
        classifier.load_state_dict(
            dist_util.load_state_dict(args.classifier_path, map_location="cpu")
        ) # check that classifier output is compatible with the number of classes
        classifier.to(dist_util.dev())
        if args.classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()

        cls_mock_out = test_classifier(classifier,args.image_size)

        assert cls_mock_out.shape[1] == args.num_classes, "the classifier has incompatible # classes"

        image_size=args.image_size
        class_cond=args.class_cond
        num_classes=args.num_classes
        num_diffusion_samples=args.num_diffusion_samples
        diffusion_sample_interval=args.diffusion_sample_interval
        specific_classes_to_sample=args.specific_classes_to_sample
    else:
        classifier=None
        image_size=None
        class_cond=None
        num_classes=None
        num_diffusion_samples=None
        diffusion_sample_interval=0
        specific_classes_to_sample=None

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        # args for diffusion sampling
        classifier=classifier,
        image_size=image_size,
        class_cond=class_cond,
        num_classes=num_classes,
        num_diffusion_samples=num_diffusion_samples,
        diffusion_sample_interval=diffusion_sample_interval,
        specific_classes_to_sample=specific_classes_to_sample
    ).run_loop()


def create_argparser():
    defaults = dict(
        # Trainer and data default args
        data_dir="",
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        lr=1e-4,
        ema_rate="0.9999", # comma-separated list of EMA values
        log_interval=10,
        save_interval=1000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler="uniform",
        weight_decay=0.0,
        lr_anneal_steps=0,

        # classifier args

        # classifier and diffusion sampling arguments
        class_subset = "default",
        class_mapping = None,
        produce_diffusion_samples=False,
        num_diffusion_samples=16,
        diffusion_sample_interval=1000,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
        clip_denoised=True,
        specific_classes_to_sample=""
    )
    d1 = model_and_diffusion_defaults()
    del d1["num_classes"]
    defaults.update(d1)
    d2 = classifier_defaults()
    del d2["num_classes"]
    defaults.update(d2)

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument("--num_classes", type=int, default=None)
    return parser


if __name__ == "__main__":
    main()
