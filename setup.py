from setuptools import setup

setup(
    name="dldf",
    version="0.1",
    packages=["dldf"],
    url="https://github.com/Osburg/dynamic_fitting",
    license="",
    author="Aaron Paul Osburg",
    author_email="aaron.osburg@meduniwien.ac.at",
    description="",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "torch==2.3.1",
        "torchvision",
        "overrides",
        "nibabel",
        "nilearn",
        "torch-cubic-spline-grids",
        "mat73",
        "typing_extensions",
        "wandb",
        "tqdm",
        "pytorch_lightning",
        "nibabel",
    ],
)
