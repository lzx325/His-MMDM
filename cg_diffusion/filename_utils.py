import os
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import cg_diffusion.image_datasets as image_datasets
import re

IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "ppm", "bmp", "pgm", "tif", "tiff", "webp"]

def parse_patch_filename(fn):
    fn_dict=dict()
    fn_noext=os.path.splitext(fn)[0]
    if fn_noext.endswith(".processed"):
        fn_dict["patch_type"]="processed"
        fn_noext=fn_noext[:-len(".processed")]
    else:
        fn_dict["patch_type"]="original"
    
    fn_parts=fn_noext.split("__")
    assert len(fn_parts)>=2
    
    fn_dict["slide_class"]=fn_parts[0]
    patch_id=fn_parts[1]
    if re.match(r'.*\..*\.',patch_id):
        fn_dict["patch_id"]=patch_id
        fn_dict["slide_id"]=".".join(patch_id.split(".")[:-1])
        fn_dict["patch_idx"],fn_dict["patch_x"],fn_dict["patch_y"]=patch_id.split(".")[-1].split("_")
        fn_dict["patch_idx"]=int(fn_dict["patch_idx"])
        fn_dict["patch_x"]=int(fn_dict["patch_x"])
        fn_dict["patch_y"]=int(fn_dict["patch_y"])
    else:
        fn_dict["patch_id"]=patch_id


    def remove_prefix(s, prefix):
        return s[len(prefix):] if s.startswith(prefix) else s
    
    if len(fn_parts)==4:
        assert fn_parts[3].startswith("encode_")
        fn_dict["encode"]=remove_prefix(fn_parts[3],"encode_")
        fn_parts=fn_parts[:-1]

    if len(fn_parts)==3:
        assert fn_parts[2].startswith("translated_") or fn_parts[2].startswith("sample_")
        if fn_parts[2].startswith("sample_"):
            fn_dict["translated_slide_class"]=remove_prefix(fn_parts[2],"sample_")
        elif fn_parts[2].startswith("translated_"):
            fn_dict["translated_slide_class"]=remove_prefix(fn_parts[2],"translated_")
        else:
            raise ValueError("invalid patch filename")
        fn_dict["patch_type"]="translated"

    elif len(fn_parts)<3:
        fn_dict["translated_slide_class"]=pd.NA
    else:
        raise ValueError("invalid patch filename")
    return fn_dict

def parse_patch_filename_from_list(fn_list):
    fn_dict=defaultdict(list)
    for fn in fn_list:
        fn_parts=parse_patch_filename(fn)
        for key in fn_parts:
            fn_dict[key].append(fn_parts[key])
        fn_dict["fn"].append(fn)
    return pd.DataFrame(fn_dict)

def parse_directory_filenames(dir_path):
    all_files = image_datasets.list_image_files_nonrecursively(dir_path)
    info_dict=defaultdict(list)
    for fp in tqdm(all_files):
        fn=os.path.basename(fp)
        fn_dict=parse_patch_filename(fn)
        for key in fn_dict:
            info_dict[key].append(fn_dict[key])
        info_dict["fn"].append(fn)
    info_df=pd.DataFrame(info_dict)
    return info_df