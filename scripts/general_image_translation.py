"""
Class-conditional image translation from one ImageNet class to another.
"""

import argparse
import os
from pathlib import Path
import itertools
from itertools import chain

import pandas as pd
import pickle as pkl

import torch as th
import torch.distributed as dist
import torch.nn.functional as F

from PIL import Image
import yaml

import cg_diffusion.train_util as train_util
import cg_diffusion.filename_utils as filename_utils
from cg_diffusion import dist_util, logger
from cg_diffusion.image_datasets import (
    load_data_with_metadata,
    load_omics_image_data_with_metadata,
    list_image_files_nonrecursively
)
from cg_diffusion.script_util import (
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
    list_subset,
    dict_subset,
    dict_content_subset,
    prepare_input,
    prepare_modification
)

from cg_diffusion import logger

def load_omics_tables(args):
    if args.model_type!="omics_unet":
        return None
    omics_spec=dict(
        genomics_table=pd.read_csv(args.genomics_table_fp,index_col=0),
        transcriptomics_table=pd.read_csv(args.transcriptomics_table_fp,index_col=0)
    )
    args.num_genomics_genes = omics_spec['genomics_table'].shape[0]
    args.num_transcriptomics_genes = omics_spec['transcriptomics_table'].shape[0]
    return omics_spec

def load_translation_spec(args):
    if args.translation_spec.endswith(".pkl"):
        with open(args.translation_spec,"rb") as f:
            translation_spec = pkl.load(f)
    elif args.translation_spec.endswith(".yaml"):
        with open(args.translation_spec,"r") as f:
            translation_spec = yaml.safe_load(f)

    return translation_spec

def get_out_filenames(extra,label):
    out_filenames = list()
    out_filepaths = list()
    assert all(0<=i<len(extra["filepath"]) for i in label.keys())
    for i in label.keys():
        for j in range(len(label[i])):
            out_filenames.append(f"{Path(extra['filepath'][i]).stem}__sample_{label[i][j]}{Path(extra['filepath'][i]).suffix}")
            out_filepaths.append(os.path.join(logger.get_dir(), "translation", out_filenames[-1]))
    
    return out_filenames, out_filepaths

def parse_class_mapping(class_mapping_str):
    mapping_dict=dict()
    for mapping in class_mapping_str.split(','):
        cl,cl_int=mapping.split(':')
        mapping_dict[cl]=int(cl_int)
    return mapping_dict

def main():
    args = create_argparser().parse_args()
    assert args.model_type in ["omics_unet","unet"]
    with open(os.path.join(logger.get_current().dir,"args.yaml"), "w") as f:
        yaml.dump(args.__dict__, f)
    dist_util.setup_dist()
    logger.configure()

    logger.log("preparing files...")
    omics_spec = load_omics_tables(args)
    if omics_spec:
        logger.log("omics spec:")
        logger.log(f"    genomics_table: {omics_spec['genomics_table'].shape}")
        logger.log(f"    transcriptomics_table: {omics_spec['transcriptomics_table'].shape}")

    translation_spec=load_translation_spec(args)
    logger.log("translation spec:")
    
    if "fn_list" in translation_spec:
        logger.log("    fn_list: {}".format(len(translation_spec['fn_list'])))
        fp_list = [os.path.join(args.data_dir,fn) for fn in translation_spec["fn_list"]]
    else:
        fp_list = [f for f in list_image_files_nonrecursively(args.data_dir) if "translated" not in f and "processed" not in f]
        logger.log("    fp_list: {}".format(len(fp_list)))

    os.makedirs(os.path.join(logger.get_dir(), "translation"), exist_ok=True)

    metadata = dict()
    fp_list_full = [fp for fp in Path(args.data_dir).glob("*.png")]
    for fp in fp_list_full:
        metadata[str(fp)] = filename_utils.parse_patch_filename(fp.name)["slide_class"]
    
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

    args.rev_class_mapping = {v: k for k, v in args.class_mapping.items()}

    if "default" in translation_spec["modifications"]:
        logger.log("    default modification:")
        logger.log(translation_spec["modifications"]["default"])

    if "custom" in translation_spec["modifications"]:
        logger.log("    custom modifications: {}".format(len(translation_spec['modifications']['custom'])))

    if args.num_classes <= 30:
        logger.log(f"num_classes: {args.num_classes}, class mapping: {args.class_mapping}")
    else:
        logger.log(f"num_classes: {args.num_classes}")

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    if args.model_type=="omics_unet":
        def model_fn(
            x, t, y, 
            genomics_genes, genomics_mutation, genomics_multiplier, 
            transcriptomics_genes, transcriptomics_exp,  transcriptomics_multiplier
        ):
            return model(
                x = x, timesteps = t, y = y,
                genomics_genes = genomics_genes,
                genomics_mutation = genomics_mutation,
                genomics_multiplier = genomics_multiplier,
                transcriptomics_genes = transcriptomics_genes,
                transcriptomics_exp = transcriptomics_exp,
                transcriptomics_multiplier = transcriptomics_multiplier
            )
    else:
        def model_fn(x, t, y):
            return model(x=x, timesteps=t, y=y)

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    def cond_fn(x, t, y, **kwargs):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale 
    
    dist.barrier()
    
    logger.log("prepare image translation datasets...")
    if args.model_type == "omics_unet":
        data = load_omics_image_data_with_metadata(
            data_dir=None,
            fp_list=[str(fp) for fp in fp_list],
            batch_size=args.batch_size,
            image_size=args.image_size,
            metadata=metadata,
            omics_spec=omics_spec,
            clstr_to_int=lambda x: args.class_mapping[x],
            filepath=True,
            deterministic=True,
            random_crop=False,
            random_flip=False,
            distributed=True,
            infinite=False,
            dataset_format=args.dataset_format
        )
    else:
        data = load_data_with_metadata(
            data_dir=None,
            fp_list=[str(fp) for fp in fp_list],
            batch_size=args.batch_size,
            image_size=args.image_size,
            metadata=metadata,
            clstr_to_int=lambda x: args.class_mapping[x],
            filepath=True,
            deterministic=True,
            random_crop=False,
            random_flip=False,
            distributed=True,
            infinite=False
        )

    dist.barrier()
    logger.log("starting image translation...")
    # some configurations
    save_intermediates=True
    save_intermediates_steps = 20
    add_multiplier=True
    if args.model_type=="omics_unet":
        tensor_keys=["y","genomics_genes","genomics_mutation","genomics_multiplier","transcriptomics_genes","transcriptomics_exp", "transcriptomics_multiplier"]
    else:
        tensor_keys=["y"]
    all_keys = tensor_keys + ["filepath"]

    for i, (batch, extra) in enumerate(data):
        logger.log(f"translating batch {i}/{len(data)}, shape {batch.shape}.")
        # saving processed images
        logger.log("saving the original, cropped images.")
        logger.log(extra["filepath"])
        images = ((batch + 1) * 127.5).clamp(0, 255).to(th.uint8)
        images = images.permute(0, 2, 3, 1)
        images = images.contiguous()
        images = images.cpu().numpy()
        for index in range(images.shape[0]):
            filepath = Path(extra["filepath"][index])
            newfilepath = os.path.join(logger.get_dir(), "translation", filepath.stem+".processed"+filepath.suffix)
            image = Image.fromarray(images[index])
            logger.log(f"    saving: {newfilepath}")
            image.save(newfilepath)
        # move to device
        batch = batch.to(dist_util.dev())
        if (args.model_type=="omics_unet") and add_multiplier:
            extra["genomics_multiplier"]=th.ones(batch.shape[0],dtype=th.float32)
            extra["transcriptomics_multiplier"]=th.ones(batch.shape[0],dtype=th.float32)

        for k in tensor_keys:
            extra[k] = extra[k].to(dist_util.dev())
        extra_tensor=dict_subset(extra,tensor_keys)
        
        # prepare modifications
        
        assert "modifications" in translation_spec
        default_modification = translation_spec["modifications"].get("default",dict())
        if "custom" in translation_spec["modifications"]:
            batch_modification = dict()
            allow_invalid_genes = list()
            for i in range(batch.shape[0]):
                batch_modification[i] = translation_spec["modifications"]["custom"].get(Path(extra["filepath"][i]).name,default_modification)
                if Path(extra["filepath"][i]).name in translation_spec["modifications"]["custom"]:
                    allow_invalid_genes.append(False)
                else:
                    allow_invalid_genes.append(True)

        modification, modification_label = prepare_modification(extra, batch_modification, args.class_mapping, omics_spec, allow_invalid_genes=allow_invalid_genes)
        out_filenames, out_filepaths = get_out_filenames(extra, modification_label)
        process_indices = list()
        
        for i, fp in enumerate(out_filepaths):
            if not os.path.isfile(fp):
                process_indices.append(i)
        
        if len(process_indices)==0:
            logger.log(f"{len(batch)} images in this batch, expecting {len(out_filepaths)} translated images in total, existing {len(out_filepaths)-len(process_indices)} translated images, skipping")
            continue
        else:
            logger.log(f"{len(batch)} images in this batch, expecting {len(out_filepaths)} translated images in total, existing {len(out_filepaths)-len(process_indices)} translated images, processing {len(process_indices)} images")
        out_filenames = list_subset(out_filenames,process_indices)
        out_filepaths = list_subset(out_filepaths,process_indices)
        # First, use DDIM to encode to latents.
        logger.log("encoding the source images.")
        if not save_intermediates:
            noise = diffusion.ddim_reverse_sample_loop(
                model_fn,
                batch,
                clip_denoised=False,
                model_kwargs=extra_tensor,
                device=dist_util.dev()
            )
        else:
            noise, diffusion_intermediates = diffusion.ddim_reverse_sample_loop(
                model_fn,
                batch,
                clip_denoised=False,
                model_kwargs=extra_tensor,
                device=dist_util.dev(),
                return_intermediates=True
            )

        logger.log(f"obtained latent representation for {batch.shape[0]} samples...")
        logger.log(f"latent with mean {noise.mean()} and std {noise.std()}")
        
        # Next, expand the source latents to the target classes.
        
        noise_expanded, extra_expanded, expand_indices = prepare_input(noise, extra, modification, return_expand_indices=True)
        noise_expanded = noise_expanded[process_indices]
        extra_expanded = dict_content_subset(extra_expanded,process_indices)
        extra_tensor_expanded=dict_subset(extra_expanded,tensor_keys)
            
        logger.log(f"to create {len(noise_expanded)} translated images...")
        # Next, decode the latents to the target class.
        sample_list = list()
        denoising_intermediates_list = None
        for i, (noise_expanded_mb, extra_tensor_expanded_mb) in enumerate(train_util.split_microbatches_with_extra(args.batch_size, noise_expanded, extra_tensor_expanded)):
            logger.log("decoding the latent representations: microbatch {}/{}".format(i+1, (len(noise_expanded)+args.batch_size-1)//args.batch_size))
            if not save_intermediates:
                sample_mb = diffusion.ddim_sample_loop(
                    model_fn,
                    (noise_expanded_mb.shape[0], 3, args.image_size, args.image_size),
                    noise=noise_expanded_mb,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=extra_tensor_expanded_mb,
                    cond_fn=cond_fn,
                    device=dist_util.dev(),
                    eta=args.eta
                )
                sample_list.append(sample_mb)
            else:
                sample_mb, denoising_intermediates_mb = diffusion.ddim_sample_loop(
                    model_fn,
                    (noise_expanded_mb.shape[0], 3, args.image_size, args.image_size),
                    noise=noise_expanded_mb,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=extra_tensor_expanded_mb,
                    cond_fn=cond_fn,
                    device=dist_util.dev(),
                    eta=args.eta,
                    return_intermediates=True
                )
                sample_list.append(sample_mb)
                if denoising_intermediates_list is None:
                    denoising_intermediates_list = [[ts] for ts in denoising_intermediates_mb]
                else:
                    for t in range(len(denoising_intermediates_list)):
                        denoising_intermediates_list[t].append(denoising_intermediates_mb[t])

        sample = th.cat(sample_list, dim=0)

        def post_process_batch_images(batch):
            batch_image = ((batch + 1) * 127.5).clamp(0, 255).to(th.uint8)
            batch_image = batch_image.permute(0, 2, 3, 1)
            batch_image = batch_image.contiguous()
            batch_image = batch_image.detach().cpu().numpy()
            return batch_image
        
        sample_arr = post_process_batch_images(sample)
        
        if save_intermediates:
            denoising_intermediates=[th.cat(denoising_intermediates_list[i],dim=0) for i in range(len(denoising_intermediates_list))]
            diffusion_intermediates_arr=dict()
            denoising_intermediates_arr=dict()
            for i in range(0,len(diffusion_intermediates),len(diffusion_intermediates)//save_intermediates_steps):
                diffusion_intermediates_arr[i]=post_process_batch_images(diffusion_intermediates[i])
            for i in range(0,len(denoising_intermediates),len(denoising_intermediates)//save_intermediates_steps):
                denoising_intermediates_arr[i]=post_process_batch_images(denoising_intermediates[i])

        logger.log("saving translated images.")
        
        for index in range(sample_arr.shape[0]):
            filename, filepath = out_filenames[index], out_filepaths[index]
            image = Image.fromarray(sample_arr[index])
            image.save(filepath)
            logger.log(f"    saving: {filepath}")

            if save_intermediates:
                # TODO: save intermediates
                out_dir = os.path.dirname(filepath)
                stem, _ = os.path.splitext(filename)
                intermediate_dir=os.path.join(out_dir, f"{stem}__intermediates")
                logger.log(f"    saving: intermediates to {intermediate_dir}")
                Path(intermediate_dir).mkdir(parents=True, exist_ok=True)
                for i in diffusion_intermediates_arr.keys():
                    image = Image.fromarray(diffusion_intermediates_arr[i][expand_indices[process_indices[index]]])
                    image.save(os.path.join(intermediate_dir,f"step01_diffusion_{i}.png"))
                for i in denoising_intermediates_arr.keys():
                    image = Image.fromarray(denoising_intermediates_arr[i][index])
                    image.save(os.path.join(intermediate_dir,f"step02_denoising_{i}.png"))
    dist.barrier()
    logger.log(f"domain translation complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        batch_size=8,
        classifier_scale=1.0,
        eta=0.0,
        class_mapping = None
    )
    d1 = model_and_diffusion_defaults()
    del d1["num_classes"]
    defaults.update(d1)

    d2 = classifier_defaults()
    del d2["num_classes"]
    defaults.update(d2)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
    )
    parser.add_argument(
        "--classifier_path",
        type=str,
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
    )

    parser.add_argument(
        "--translation_spec",
        type=str,
        required=True,
        help = "config for translation"
    )

    parser.add_argument(
        "--genomics_table_fp",
        type=str,
        default=None
    )

    parser.add_argument(
        "--transcriptomics_table_fp",
        type=str,
        default=None
    )

    parser.add_argument(
        "--dataset_format",
        type = str,
        default = "TCGA"
    )

    parser.add_argument("--num_classes", type=int, default=None) # make sure the the parsed type is int

    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
    if False:
        args = argparse.Namespace()
        args.model_type="omics_unet"
        args.genomics_table_fp="downloaded_datasets/curated/genomics_table.csv"
        args.transcriptomics_table_fp="downloaded_datasets/curated/transcriptomics_table.csv"

        inferred_class_mapping={'TCGA-COAD': 0, 'TCGA-ESCA': 1, 'TCGA-LUAD': 2, 'TCGA-LUSC': 3, 'TCGA-READ': 4, 'TCGA-STAD': 5}
        omics_spec=load_omics_tables(args)

        bs = 8
        batch = th.rand(bs,3,128,128)
        y=th.tensor([0,1,2,3,4,5,0,1],dtype=th.int64)
        genomics_genes=th.tensor(
            [
                [12,13,14,247],
                [0,10,11,12],
                [0,10,11,12],
                [0,10,11,12],
                [0,10,11,12],
                [0,10,11,12],
                [0,10,11,12],
                [0,10,14,15],
            ],
            dtype=th.int64
        )
        genomics_muts=th.tensor(
            [[0,0,0,0],[1,1,1,2]]+
            [[0,0,0,0]]*(bs-2),
            dtype=th.int64
        )
        genomics_multiplier=th.ones(bs,dtype=th.float32)
        transcriptomics_genes=th.tensor(
            [[2087,2088,2089,2090]]*bs,
            dtype=th.int64
        )
        transcriptomics_exp=th.tensor(
            [[0.5,0.5,0.5,0.5]]*bs,
            dtype=th.float32
        )
        transcriptomics_multiplier=th.ones(bs,dtype=th.float32)
        extra = dict(
            y=y,
            genomics_genes=genomics_genes,
            genomics_mutation=genomics_muts,
            genomics_multiplier=genomics_multiplier,
            transcriptomics_genes=transcriptomics_genes,
            transcriptomics_exp=transcriptomics_exp,
            transcriptomics_multiplier=transcriptomics_multiplier,
            filepath=[f"image__{i}.png" for i in range(bs)]
        )
        modification1 = {
            0: {
                "contents":{
                    "y":["TCGA-LUAD","TCGA-LUAD","TCGA-LUSC"],
                    "genomics_mutation":{
                        "TP53":[1,2,None],
                        "SMAD4":[1,1,None],
                    },
                    "genomics_multiplier":[0.5,1,2],
                    "transcriptomics_exp":{
                        "TTK":[0,0.5,1]
                    },
                    "transcriptomics_multiplier":[1,3,10]
                },
                "labels":["mutation1","mutation2","mutation3"]
                    
            },
            1: {
                "contents":{
                    "y":["TCGA-LUAD"]
                },
                "labels":["LUAD"]
                
            }
        }

        modification = {
            0: {
                "contents":{
                    "y": [0,1,2,3]
                },
                "labels":["class0","class1","class2","class3"]
            }
        }


        modification, label = prepare_modification(
            extra,modification,
            class_mapping=inferred_class_mapping,
            omics_spec=omics_spec
        )

        out_filenames,out_filepaths=get_out_filenames(extra,label)
        batch_expanded, extra_expanded = prepare_input(batch,extra,modification)

