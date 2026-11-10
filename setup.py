from setuptools import setup, find_packages

setup(
    name="semi_flood",
    version="1.0.0",
    description="A Reproducible Semi-Supervised Flood Segmentation Benchmark for Aerial & UAV Imagery",
    author="Semi-Flood Benchmark Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.22.0",
        "scipy>=1.9.0",
        "opencv-python>=4.7.0",
        "albumentations>=1.3.0",
        "timm>=0.9.0",
        "gdown>=4.7.0",
        "tqdm>=4.64.0",
        "tabulate>=0.9.0",
        "matplotlib>=3.6.0",
        "ptflops>=0.7.0",
    ],
)
