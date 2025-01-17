
# Model Card for His-MMDM

Histopathological image Multi-domain Multi-omics translation with Diffusion Models (His-MMDM) is a framework for the translation of histopathological images. Based on the principle of diffusion models, His-MMDM achieves image translation by diffusing the source domain images into noisy images distributed as Gaussian distribution and then denoising the noisy images back into target domain images. Compared with previous image translation models in histopathology, His-MMDM stands out with two unique capabilities: (1) it can be efficiently trained to translate images between an unlimited number of categorial domains (through class-conditional inputs to the model) (2) it can perform genomics- or transcriptomics- guided editing of histopathological images (through multi-omics inputs to the model). 

## Model Details

### Model Description


- **Developed by:** Li, Zhongxiao et al.
- **Funded by:** King Abdullah University of Science and Technology (KAUST) Office of Research Administration (ORA) under Award No REI/1/5234-01-01, REI/1/5414-01-01, RGC/3/4816-01-01, REI/1/5289-01-01, REI/1/5404-01-01, REI/1/5992-01-01, and URF/1/4663-01-01.
- **Model type:** Diffusion model
- **License:** MIT License

### Model Sources 

- **Repository:** https://github.com/lzx325/His-MMDM
- **Paper:** https://www.medrxiv.org/content/10.1101/2024.07.11.24310294v2

## Uses


### Direct Use

His-MMDM is intended to be used for machine learning research in computational pathology under the MIT License.

### Downstream Use

The fine-tuning of His-MMDM pretrained checkpoints is allowed as long as the process is compliant with the applicable laws or regulations in the jurisdiction.

## How to Get Started with the Model

Please refer to the [README of the His-MMDM repository](https://github.com/lzx325/His-MMDM/blob/main/README.md).

## Technical Specifications 


#### Hardware

Inference of His-MMDM requires 1-2 NVIDIA V100 GPUs. Training His-MMDM from scratch utilized 24 NVIDIA V100 GPUs (8 GPUs per node, 3 nodes in total).

#### Software

The software environment is specified in [README](https://github.com/lzx325/His-MMDM/blob/main/README.md).

## Citation

```
Zhongxiao Li et al.,"His-MMDM: Multi-domain and Multi-omics Translation of Histopathology Images with Diffusion Models"
```

## Model Card Authors 

Zhongxiao Li

## Model Card Contact

Zhongxiao Li (lzx325@outlook.com)
