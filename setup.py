from setuptools import setup, find_packages

setup(
    name="His-MMDM",
    version="0.1.0",  # Adjust as per your versioning
    author="Zhongxiao Li",
    author_email="lzx325@outlook.com",
    description="Multi-domain and Multi-omics Translation of Histopathology Images with Diffusion Models",
    keywords="Image Generative Models, Image Translation, Diffusion Models, Histopathology ",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/lzx325/His-MMDM",
    packages=["cg_diffusion"],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha"
    ],
    # conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch
    python_requires=">=3.10",
    install_requires=[
        "PyYAML==6.0.1",
        "scikit-learn==1.3.1",
        "requests==2.25",
        "tqdm",
        "blobfile",
        "tensorboardX"
    ],
    extras_require={
        "dev": [
            "pytest>=3.7",
            "check-manifest",
            "twine",
        ],
    },
)