import os
import argparse
from cg_diffusion.download_utils import download_if_not_exist, extract_tar_gz_with_progress

REPO_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_remote_files_manifest(remote_prefix, local_prefix, fp_list):
    remote_prefix = remote_prefix.rstrip("/")
    local_prefix = local_prefix.rstrip("/")
    fp_list = [fp.rstrip("/") for fp in fp_list]
    remote_files_manifest = []
    for fp in fp_list:
        remote_fp = os.path.join(remote_prefix, fp)
        local_fp = os.path.join(local_prefix, fp)
        remote_files_manifest.append((local_fp, remote_fp))
    return remote_files_manifest


TRAIN_FILES_CHECKSUM = {
    "train/train_demo_TCGA.tar.gz":"21c554f3d68c188a667d0023bb133873",
}

INFERENCE_FILES_CHECKSUM={
    "inference/pretrained_checkpoints.tar.gz":"7354dfba1481afea9af0672476a04ca8",
    "inference/demo_data.tar.gz":"d3f116d9e4cec7ef88c360c94f782de7"
}

local_prefix = os.path.join(REPO_DIR,"download")
remote_prefix = "https://hismmdm.s3.amazonaws.com/"

TRAIN_FILES_MANIFEST = dict(
    zip(
        TRAIN_FILES_CHECKSUM.keys(),
        get_remote_files_manifest(
            remote_prefix=remote_prefix,
            local_prefix=local_prefix,
            fp_list=list(TRAIN_FILES_CHECKSUM.keys()),
        ),
    )
)

INFERENCE_FILES_MANIFEST = dict(
    zip(
        INFERENCE_FILES_CHECKSUM.keys(),
        get_remote_files_manifest(
            remote_prefix=remote_prefix,
            local_prefix=local_prefix,
            fp_list=list(INFERENCE_FILES_CHECKSUM.keys()),
        ),
    )

)


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("dataset",type=str)
    return args.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.dataset == "train":
        for k, (local_fp, remote_fp) in TRAIN_FILES_MANIFEST.items():
            checksum = TRAIN_FILES_CHECKSUM[k]
            download_if_not_exist(local_fp, remote_fp, md5sum=checksum)
            extract_tar_gz_with_progress(local_fp)
    elif args.dataset == "inference":
        for k, (local_fp, remote_fp) in INFERENCE_FILES_MANIFEST.items():
            checksum = INFERENCE_FILES_CHECKSUM[k]
            download_if_not_exist(local_fp, remote_fp, md5sum=checksum)
            extract_tar_gz_with_progress(local_fp)