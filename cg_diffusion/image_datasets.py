from collections import defaultdict
import os
import math
import random

import pandas as pd
from PIL import Image

import numpy as np
from cg_diffusion import logger
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from pathlib import Path

import cg_diffusion.filename_utils as filename_utils

def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    recursive=False,
    deterministic=False,
    random_crop=False,
    random_flip=False, # changed
    clstr_to_int=None,
    balance=False,
    distributed=False,
    infinite= False # changed
):
    if not data_dir:
        raise ValueError("unspecified data directory")
    if recursive:
        all_files = list_image_files_recursively(data_dir)
    else:
        all_files = list_image_files_nonrecursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [os.path.basename(path).split("__")[0] for path in all_files]
        if clstr_to_int is None:
            class_mapping = infer_class_mapping(class_names)
            if len(class_mapping)<20:
                logger.log("use inferred class mapping of {} classes: {}".format(len(class_mapping),class_mapping))
            else:
                logger.log("use inferred class mapping of {} classes".format(len(class_mapping)))
            clstr_to_int = lambda x: class_mapping[x]
        classes = [int(clstr_to_int(x)) for x in class_names]

    if distributed:
        from mpi4py import MPI
        dataset = ImageDataset(
            image_size,
            all_files,
            classes=classes,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            random_crop=random_crop,
            random_flip=random_flip,
        )
    else:
        dataset = ImageDataset(
            image_size,
            all_files,
            classes=classes,
            shard=0,
            num_shards=1,
            random_crop=random_crop,
            random_flip=random_flip,
        )

    loader = get_dataloader(
        dataset = dataset,
        classes = classes,
        batch_size = batch_size,
        deterministic = deterministic,
        balance = balance
    )

    if not infinite:
        return loader
    else:
        return infinite_dataloader(loader)

def infinite_dataloader(loader):
    while True:
        yield from loader

def load_data_with_metadata(
    *,
    data_dir,
    batch_size,
    image_size,
    metadata,
    recursive=False,
    deterministic=False,
    fp_list=None,
    fn_list=None,
    random_crop=False,
    random_flip=False, # changed
    metadata_key_parser=None,
    clstr_to_int=None,
    balance=False,
    distributed=False,  # changed
    infinite=False, # change
    filepath=False, # changed
):
    if metadata_key_parser is None:
        metadata_key_parser = lambda x: x

    if fp_list is not None:
        all_files = fp_list
        keys = [metadata_key_parser(path) for path in fp_list]
    elif fn_list is not None:
        all_files = [os.path.join(data_dir,fn) for fn in fn_list]
        keys = [metadata_key_parser(path) for path in fn_list]
    else:
        if recursive:
            all_files = list_image_files_recursively(data_dir)
        else:
            all_files = list_image_files_nonrecursively(data_dir)
        keys = [metadata_key_parser(path) for path in all_files]

    # get classes from metadata
    assert set(keys).issubset(set(metadata.keys()))
    classes_str = [metadata[key] for key in keys]

    if clstr_to_int is None:
        class_mapping = {x: i for i, x in enumerate(sorted(set(classes_str)))}
        if len(class_mapping)<20:
            logger.log("use inferred class mapping of {} classes: {}".format(len(class_mapping),class_mapping))
        else:
            logger.log("use inferred class mapping of {} classes".format(len(class_mapping)))
        clstr_to_int = lambda x: class_mapping[x]

    classes = [int(clstr_to_int(x)) for x in classes_str]

    if distributed:
        from mpi4py import MPI
        dataset = ImageDataset(
            image_size,
            all_files,
            classes=classes,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            random_crop=random_crop,
            random_flip=random_flip,
            filepath=filepath
        )
    else:
        dataset = ImageDataset(
            image_size,
            all_files,
            classes=classes,
            shard=0,
            num_shards=1,
            random_crop=random_crop,
            random_flip=random_flip,
            filepath=filepath
        )

    loader = get_dataloader(
        dataset = dataset,
        classes = classes,
        batch_size = batch_size,
        deterministic = deterministic,
        balance=balance
    )
    if not infinite:
        return loader
    else:
        return infinite_dataloader(loader)

def TCGA_sample_id_parser(fn):
    return filename_utils.parse_patch_filename(fn)["slide_id"][:15]

def default_sample_id_parser(fn):
    return filename_utils.parse_patch_filename(fn)["patch_id"]

def display_stats(all_files,genomics_table,transcriptomics_table,sample_id_parser="TCGA"):
    if sample_id_parser=="TCGA":
        sample_id_parser=TCGA_sample_id_parser
    elif sample_id_parser=="default":
        sample_id_parser=default_sample_id_parser
    patch_stats_df_dict=defaultdict(list)

    for fp in all_files:
        fn=os.path.basename(fp)
        fn_dict=filename_utils.parse_patch_filename(fn)
        slide_class=fn_dict["slide_class"]
        sample_id_no_vial=sample_id_parser(fn)

        patch_stats_df_dict["fp"].append(fp)
        patch_stats_df_dict["sample_id"].append(sample_id_no_vial)
        patch_stats_df_dict["slide_class"].append(slide_class)

    patch_stats_df=pd.DataFrame(patch_stats_df_dict)

    for slide_class in sorted(patch_stats_df["slide_class"].unique()):
        samples_set=patch_stats_df.loc[patch_stats_df["slide_class"]==slide_class,"sample_id"].unique()
        samples_with_genomics=np.intersect1d(samples_set,genomics_table.columns)
        n_patches_all=patch_stats_df[patch_stats_df["sample_id"].isin(samples_set)].shape[0]
        n_patches_with_genomics=patch_stats_df[patch_stats_df["sample_id"].isin(samples_with_genomics)].shape[0]
        n_samples_with_genomics=len(samples_with_genomics)
        samples_with_transcriptomics=np.intersect1d(samples_set,transcriptomics_table.columns)
        n_patches_with_transcriptomics=patch_stats_df[patch_stats_df["sample_id"].isin(samples_with_transcriptomics)].shape[0]
        n_samples_with_transcriptomics=len(samples_with_transcriptomics)
        logger.log(
            "slide_class: {}, samples with genomics: {}/{}, patches with genomics: {}/{}\nsamples with transcriptomics: {}/{}, patches with transcriptomics: {}/{}".format(
                slide_class,
                n_samples_with_genomics, len(samples_set),
                n_patches_with_genomics, n_patches_all,
                n_samples_with_transcriptomics, len(samples_set),
                n_patches_with_transcriptomics, n_patches_all
            )
        )

def load_omics_image_data_with_metadata(
    *,
    data_dir,
    batch_size,
    image_size,
    metadata,
    omics_spec,
    recursive=False,
    deterministic=False,
    fp_list=None,
    fn_list=None,
    random_crop=False,
    random_flip=False, # changed
    metadata_key_parser=None,
    clstr_to_int=None,
    balance=False,
    distributed=False, # changed
    infinite=False, # changed
    filepath=False,
    dataset_format = "TCGA"
):
    if metadata_key_parser is None:
        metadata_key_parser = lambda x: x

    if fp_list is not None:
        all_files = fp_list
        keys = [metadata_key_parser(path) for path in fp_list]
    elif fn_list is not None:
        all_files = [os.path.join(data_dir,fn) for fn in fn_list]
        keys = [metadata_key_parser(path) for path in fn_list]
    else:
        if recursive:
            all_files = list_image_files_recursively(data_dir)
        else:
            all_files = list_image_files_nonrecursively(data_dir)
        keys = [metadata_key_parser(path) for path in all_files]


    assert set(keys).issubset(set(metadata.keys()))
    classes_str = [metadata[key] for key in keys]
    if clstr_to_int is None:
        class_mapping = infer_class_mapping(classes_str)
        if len(class_mapping)<20:
            logger.log("use inferred class mapping of {} classes: {}".format(len(class_mapping),class_mapping))
        else:
            logger.log("use inferred class mapping of {} classes".format(len(class_mapping)))
        clstr_to_int = lambda x: class_mapping[x]
    classes = [int(clstr_to_int(x)) for x in classes_str]

    genomics_table=omics_spec["genomics_table"]
    transcriptomics_table=omics_spec["transcriptomics_table"]
    display_stats(all_files,genomics_table,transcriptomics_table,sample_id_parser=dataset_format)
    
    def image_path_to_omics(path,sample_id_parser="TCGA",notfound='placeholder'):
        if sample_id_parser=="TCGA":
            sample_id_parser=TCGA_sample_id_parser
        elif sample_id_parser=="default":
            sample_id_parser=default_sample_id_parser

        fn=os.path.basename(path)
        fn_dict=filename_utils.parse_patch_filename(fn)

        sample_id_no_vial=sample_id_parser(fn)

        genomics_genes=list(genomics_table.index)
        genomics_genes_idx=[genomics_table.index.get_loc(i) for i in genomics_genes]
        if sample_id_no_vial in genomics_table.columns:
            genomics_data=genomics_table.loc[genomics_genes,sample_id_no_vial].values.astype(np.int64)
        else:
            if notfound=='placeholder':
                genomics_data=np.full((len(genomics_genes),),2,dtype=np.int64)
            elif notfound=="raise":
                raise ValueError("genomics data not found for {}".format(sample_id_no_vial))

        transcriptomics_genes=list(transcriptomics_table.index)
        transcriptomics_genes_idx=[transcriptomics_table.index.get_loc(i) for i in transcriptomics_genes]
        if sample_id_no_vial in transcriptomics_table.columns:
            transcriptomics_data=transcriptomics_table.loc[transcriptomics_genes,sample_id_no_vial].values.astype(np.float32)
        else:
            if notfound=='placeholder':
                transcriptomics_data=np.full((len(transcriptomics_genes),),np.nan,dtype=np.float32)
            elif notfound=="raise":
                raise ValueError("transcriptomics data not found for {}".format(sample_id_no_vial))

        return {
            "genomics_genes":np.array(genomics_genes_idx,dtype=np.int64),
            "genomics_mutation":genomics_data,
            "transcriptomics_genes":np.array(transcriptomics_genes_idx,dtype=np.int64),
            "transcriptomics_exp":transcriptomics_data,
        }
    
    if distributed:
        from mpi4py import MPI
        dataset = OmicsImageDataset(
            image_size,
            all_files,
            image_path_to_omics=lambda x: image_path_to_omics(x,sample_id_parser="TCGA") if dataset_format=="TCGA" else image_path_to_omics(x,sample_id_parser="default"),
            classes=classes,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            random_crop=random_crop,
            random_flip=random_flip,
            filepath=filepath
        )
    else:
        dataset = OmicsImageDataset(
            image_size,
            all_files,
            image_path_to_omics=lambda x: image_path_to_omics(x,sample_id_parser="TCGA") if dataset_format=="TCGA" else image_path_to_omics(x,sample_id_parser="default"),
            classes=classes,
            shard = 0,
            num_shards = 1,
            random_crop=random_crop,
            random_flip=random_flip,
            filepath=filepath
        )

    loader = get_dataloader(
        dataset = dataset,
        classes = classes,
        batch_size = batch_size,
        deterministic = deterministic,
        balance=balance
    )

    if not infinite:
        return loader
    else:
        return infinite_dataloader(loader)

def get_dataloader(dataset, classes, batch_size, deterministic=False, balance=False):
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, sampler = None, shuffle=False, num_workers=1, drop_last=False
        )
    else:
        if balance:
            assert classes is not None
            _,classes_count=np.unique(classes,return_counts=True)
            if len(classes_count)<20:
                logger.log("Using weighted sampling, the sample weights are {}".format(1/classes_count))
            else:
                logger.log("Using weighted sampling, first 20 weights are {}".format(1/classes_count[:20]))
            sample_weight=[1/classes_count[i] for i in dataset.local_classes]
            sampler=WeightedRandomSampler(sample_weight,len(dataset))
            loader = DataLoader(
                dataset, batch_size=batch_size, sampler = sampler, num_workers=1, drop_last=True
            )
        else:
            loader = DataLoader(
                dataset, batch_size=batch_size, sampler = None, shuffle=True, num_workers=1, drop_last=True
            )

    return loader

def _list_image_files_recursively(data_dir):
    from .filename_utils import IMAGE_EXTENSIONS
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in IMAGE_EXTENSIONS:
            results.append(full_path)
        elif os.path.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

def list_image_files_recursively(data_dir):
    return _list_image_files_recursively(data_dir)

def list_image_files_nonrecursively(data_dir):
    """List images files in the directory (not recursively)."""
    from .filename_utils import IMAGE_EXTENSIONS
    files = sorted(os.listdir(data_dir))
    results = []
    for entry in files:
        full_path = os.path.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in IMAGE_EXTENSIONS:
            results.append(full_path)
    return results

def infer_class_mapping(cl_list):
    class_mapping = {x: i for i, x in enumerate(sorted(set(cl_list)))}
    return class_mapping

class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        filepath=False
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.filepath=filepath


    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with open(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        if self.filepath:
            out_dict["filepath"] = self.local_images[idx]

        return np.transpose(arr, [2, 0, 1]), out_dict

class OmicsImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        image_path_to_omics,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        filepath=False
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.filepath=filepath
        self.image_path_to_omics=image_path_to_omics

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with open(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        omics_data=self.image_path_to_omics(path)
        for k,v in omics_data.items():
            out_dict[k] = v
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        if self.filepath:
            out_dict["filepath"] = path
        return np.transpose(arr, [2, 0, 1]), out_dict

def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

def get_image_filenames_for_label(label):
    """
    Returns the validation files for images with the given label. This is a utility
    function for ImageNet translation experiments.
    :param label: an integer in 0-1000
    """
    # First, retrieve the synset word corresponding to the given label
    base_dir = os.getcwd()
    synsets_filepath = os.path.join(base_dir, "evaluations", "synset_words.txt")
    synsets = [line.split()[0] for line in open(synsets_filepath).readlines()]
    synset_word_for_label = synsets[label]

    # Next, build the synset to ID mapping
    synset_mapping_filepath = os.path.join(base_dir, "evaluations", "map_clsloc.txt")
    synset_to_id = dict()
    with open(synset_mapping_filepath) as file:
        for line in file:
            synset, class_id, _ = line.split()
            synset_to_id[synset.strip()] = int(class_id.strip())
    true_label = synset_to_id[synset_word_for_label]

    # Finally, return image files corresponding to the true label
    validation_ground_truth_filepath = os.path.join(base_dir, "evaluations", "ILSVRC2012_validation_ground_truth.txt")
    source_data_labels = [int(line.strip()) for line in open(validation_ground_truth_filepath).readlines()]
    image_indexes = [i + 1 for i in range(len(source_data_labels)) if true_label == source_data_labels[i]]
    output = [f"ILSVRC2012_val_{str(i).zfill(8)}.JPEG" for i in image_indexes]
    return output



if __name__=="__main__":
    pass

