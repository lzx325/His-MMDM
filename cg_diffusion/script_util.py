import argparse
import inspect
import itertools 
from PIL import Image
import torch as th
from torchvision import models, transforms

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet import SuperResModel, UNetModel, OmicsUNetModel, EncoderUNetModel

NUM_CLASSES = 1000


def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict( # 8 args
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )


def classifier_defaults():
    """
    Defaults for classifier models.
    """
    return dict(
        image_size=64,
        classifier_use_fp16=False,
        classifier_width=128,
        classifier_depth=2,
        classifier_attention_resolutions="32,16,8",  # 16
        classifier_use_scale_shift_norm=True,  # False
        classifier_resblock_updown=True,  # False
        classifier_pool="attention",
        num_classes=NUM_CLASSES
    )


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        model_type="unet",
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        num_classes=NUM_CLASSES,
        num_genomics_genes=None,
        num_transcriptomics_genes=None,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
    )
    res.update(diffusion_defaults())
    return res


def classifier_and_diffusion_defaults():
    res = classifier_defaults()
    res.update(diffusion_defaults())
    return res


def create_model_and_diffusion(
    image_size,
    class_cond,
    num_classes,
    num_genomics_genes,
    num_transcriptomics_genes,
    learn_sigma,
    num_channels,
    num_res_blocks,
    model_type,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        model_type=model_type,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        num_classes=num_classes,
        num_genomics_genes=num_genomics_genes,
        num_transcriptomics_genes=num_transcriptomics_genes,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion

# 17 args
def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    model_type="unet",
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    num_classes=NUM_CLASSES,
    num_genomics_genes=None,
    num_transcriptomics_genes=None,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    if model_type == "unet":
        return UNetModel(
            image_size=image_size,
            in_channels=3,
            model_channels=num_channels,
            out_channels=(3 if not learn_sigma else 6),
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            num_classes=(num_classes if class_cond else None),
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
        )
    elif model_type == "omics_unet":
        return OmicsUNetModel(
            image_size=image_size,
            in_channels=3,
            model_channels=num_channels,
            out_channels=(3 if not learn_sigma else 6),
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            num_classes=(num_classes if class_cond else None),
            num_genomics_genes=num_genomics_genes,
            num_transcriptomics_genes=num_transcriptomics_genes,
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
        )
    else:
        raise ValueError(f"unknown model type: {model_type}")


def create_classifier_and_diffusion(
    image_size,
    classifier_use_fp16,
    classifier_width,
    classifier_depth,
    classifier_attention_resolutions,
    classifier_use_scale_shift_norm,
    classifier_resblock_updown,
    classifier_pool,
    num_classes,
    learn_sigma,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
):
    classifier = create_classifier(
        image_size,
        classifier_use_fp16,
        classifier_width,
        classifier_depth,
        classifier_attention_resolutions,
        classifier_use_scale_shift_norm,
        classifier_resblock_updown,
        classifier_pool,
        num_classes=num_classes
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return classifier, diffusion

def modified_resnet18(weights, num_classes=NUM_CLASSES):
    import torch.nn as nn
    from torchvision.models import resnet18
    # Load the pre-trained model
    model = resnet18(weights=weights)

    # Modify the first convolution layer to accept an input size of 128x128
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # If you're working on a classification problem with a different number of classes,
    # you may need to modify the number of output neurons in the final layer.
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

# for FID computation
def inceptionv3():
    inception_v3 = models.inception_v3(pretrained=True)
    inception_v3.fc = th.nn.Identity()
    inception_v3.AuxLogits.fc = th.nn.Identity()
    inception_v3.fc = th.nn.Identity()
    inception_v3.AuxLogits.fc = th.nn.Identity()
    return inception_v3

class InceptionV3Dataset(th.utils.data.Dataset):
    def __init__(self, fp_list):
        self.fp_list = fp_list
        self.preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.fp_list)

    def __getitem__(self, idx):
        
        img = Image.open(self.fp_list[idx])
        img = self.preprocess(img)
        return img
    
def extract_features(model, dataset, batch_size=32):
    data_loader = th.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    device = next(model.parameters()).device
    model.eval()  # Set the model to evaluation mode
    features = []
    with th.no_grad():  # No need to track gradients
        for inputs in data_loader:
            inputs = inputs.to(device)
            output = model(inputs).detach().cpu()
            features.append(output)
    
    features = th.cat(features, dim=0).numpy()
    return features

# 9 args
def create_classifier(
    image_size,
    classifier_use_fp16,
    classifier_width,
    classifier_depth,
    classifier_attention_resolutions,
    classifier_use_scale_shift_norm,
    classifier_resblock_updown,
    classifier_pool,
    num_classes=NUM_CLASSES,
):
    if image_size == 512:
        channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
    elif image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4) # lizx: channel multiplier for each block
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in classifier_attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return EncoderUNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=classifier_width,
        out_channels=num_classes,
        num_res_blocks=classifier_depth,
        attention_resolutions=tuple(attention_ds),
        channel_mult=channel_mult,
        use_fp16=classifier_use_fp16,
        num_head_channels=64,
        use_scale_shift_norm=classifier_use_scale_shift_norm,
        resblock_updown=classifier_resblock_updown,
        pool=classifier_pool,
    )


def sr_model_and_diffusion_defaults():
    res = model_and_diffusion_defaults()
    res["large_size"] = 256
    res["small_size"] = 64
    arg_names = inspect.getfullargspec(sr_create_model_and_diffusion)[0]
    for k in res.copy().keys():
        if k not in arg_names:
            del res[k]
    return res


def sr_create_model_and_diffusion(
    large_size,
    small_size,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
):
    model = sr_create_model(
        large_size,
        small_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def sr_create_model(
    large_size,
    small_size,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    resblock_updown,
    use_fp16,
    num_classes 
):
    _ = small_size  # hack to prevent unused variable

    if large_size == 512:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif large_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif large_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported large size: {large_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(large_size // int(res))

    return SuperResModel(
        image_size=large_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(num_classes if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
    )

# 9 args
def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def dict_subset(d,keys):
    return {k:d[k] for k in keys}

def list_subset(l,indices):
    return [l[i] for i in indices]

def dict_content_subset(d,indices):
    d_expanded = dict()
    for k in d.keys():
        if isinstance(d[k],list):
            d_expanded[k]=list_subset(d[k],indices)
        else:
            d_expanded[k]=d[k][indices]
    return d_expanded

def prepare_modification(extra,modification,class_mapping,omics_spec=None,device=None, allow_invalid_genes=False):
    import torch as th
    if device is None:
        device=next(iter(extra.values())).device

    if isinstance(allow_invalid_genes,bool):
        allow_invalid_genes=[allow_invalid_genes]*len(modification)
    elif isinstance(allow_invalid_genes,list):
        assert len(allow_invalid_genes)==len(modification)

    modification_dict=dict()
    label_dict=dict()
    for i, dct in modification.items():
        # processing modification contents
        mod=dct["contents"]
        modification_dict[i] = dict()
        bs = None
        for k, v in mod.items():
            if k=="y":
                assert isinstance(v,list) and (bs is None or len(v)==bs)
                bs = len(v)
                cl_list = list()
                for cl in v:
                    if cl is None: # no modification
                        cl = extra["y"][i]
                    elif type(cl)!=int:
                        cl = class_mapping[cl]
                    cl_list.append(cl)
                modification_dict[i]["y"]=th.tensor(cl_list,dtype=th.int64)
            if k in ["genomics_multiplier","transcriptomics_multiplier"]:
                assert isinstance(v,list) and (bs is None or len(v)==bs)
                bs = len(v)
                mult_list = list()
                for mult in v:
                    if mult is None: # no modification
                        mult = extra[k][i]
                    mult_list.append(mult)
                modification_dict[i][k]=th.tensor(mult_list,dtype=th.float32)

            elif k == "genomics_mutation":
                assert isinstance(v,dict) and omics_spec is not None

                # check validity
                for mut_gene, mut_values in v.items():
                    assert isinstance(mut_values,list)
                    if bs is None:
                        bs = len(mut_values)
                    else:
                        assert len(mut_values)==bs

                mut_tensor_list=list()
                for mut_idx in range(bs):
                    unmut_tensor = extra["genomics_mutation"][i].clone()
                    mut_tensor = unmut_tensor.clone()
                    for mut_gene_ in v.keys():
                        if type(mut_gene_)!=int:
                            mut_gene = omics_spec["genomics_table"].index.get_loc(mut_gene_)
                        else:
                            mut_gene = mut_gene_
                        try:
                            mut_gene_idx = extra["genomics_genes"][i].tolist().index(mut_gene)
                        except ValueError:
                            if allow_invalid_genes[i]:
                                continue
                            else:
                                raise ValueError(f"invalid gene {mut_gene_} for batch index {i}")
                        mut_value = v[mut_gene_][mut_idx]
                        if mut_value is None:
                            mut_value = unmut_tensor[mut_gene_idx]
                        mut_tensor[mut_gene_idx]=mut_value
                    mut_tensor_list.append(mut_tensor)
                mut_tensor = th.stack(mut_tensor_list)
                modification_dict[i]["genomics_mutation"]=mut_tensor

            elif k == "transcriptomics_exp":
                assert isinstance(v,dict) and omics_spec is not None

                # check validity
                for exp_gene, exp_values in v.items():
                    assert isinstance(exp_values,list)
                    if bs is None:
                        bs = len(exp_values)
                    else:
                        assert len(exp_values)==bs

                exp_tensor_list=list()
                for exp_idx in range(bs):
                    original_exp_tensor = extra["transcriptomics_exp"][i].clone()
                    new_exp_tensor = original_exp_tensor.clone()
                    for exp_gene_ in v.keys():
                        if type(exp_gene_)!=int:
                            exp_gene = omics_spec["transcriptomics_table"].index.get_loc(exp_gene_)
                        else:
                            exp_gene = exp_gene_
                        try:
                            exp_gene_idx = extra["transcriptomics_genes"][i].tolist().index(exp_gene)
                        except ValueError:
                            if allow_invalid_genes[i]:
                                continue
                            else:
                                raise ValueError(f"invalid gene {exp_gene_} for batch index {i}")
                        exp_value = v[exp_gene_][exp_idx]
                        if exp_value is None:
                            exp_value = original_exp_tensor[exp_gene_idx]
                        new_exp_tensor[exp_gene_idx]=exp_value
                    exp_tensor_list.append(new_exp_tensor)

                exp_tensor = th.stack(exp_tensor_list)
                modification_dict[i]["transcriptomics_exp"]=exp_tensor
        # processing modification labels
        if "labels" in dct:
            lab = dct["labels"]
            assert isinstance(lab,list) and (len(lab)==bs or bs is None)
        else:
            lab = list(range(bs))

        label_dict[i]=lab
    return modification_dict, label_dict

def prepare_input(batch,extra,modification,device=None,cartesian_product=False, return_expand_indices=False):
    import torch as th
    from math import prod
    device=batch.device if device is None else device
    bs = batch.shape[0]
    for v in extra.values():
        assert (
            (isinstance(v,th.Tensor) and v.shape[0]==bs)
            or (isinstance(v,list) and len(v)==bs)
        )
    for k,v in modification.items():
        assert k<bs and isinstance(v,dict) 
        for k2,v2 in v.items():
            assert isinstance(v2,th.Tensor)

    expand_indices=list()
    for i in range(bs):
        if i in modification and len(modification[i])>0:
            if cartesian_product:
                expand_indices.extend([i]*prod([len(modification[i][k]) for k in modification[i].keys()]))
            else:
                mod_len=len(next(iter(modification[i].values())))

                assert all(len(v)==mod_len for v in modification[i].values())
                expand_indices.extend([i]*mod_len)
        else:
            pass
    batch_expanded = batch[expand_indices].to(device)
    extra_expanded = dict_content_subset(extra,expand_indices)

    batch_idx = 0
    for i in range(bs):
        if i in modification and len(modification[i])>0:
            mod_keys=list(modification[i].keys())
            n_muts_d = {k: len(modification[i][k]) for k in mod_keys}
            if cartesian_product:
                for indices in itertools.product(*[range(n_muts_d[k]) for k in mod_keys]):
                    for j in range(len(mod_keys)):
                        extra_expanded[mod_keys[j]][batch_idx]=modification[i][mod_keys[j]][indices[j]].to(device)
                    batch_idx += 1
            else:
                for j in range(len(mod_keys)):
                    extra_expanded[mod_keys[j]][batch_idx:batch_idx+n_muts_d[mod_keys[j]]]=modification[i][mod_keys[j]].to(device)
                batch_idx += n_muts_d[mod_keys[0]]
        else:
            pass
    if return_expand_indices:
        return batch_expanded,extra_expanded,expand_indices
    else:
        return batch_expanded,extra_expanded
    
if __name__=="__main__":
    pass
