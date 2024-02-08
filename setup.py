from setuptools import setup, find_packages

setup(
    name='bcell',
    version='1.0',
    description='Bcell epitope predict tool',
    packages=find_packages(),
    install_requires=[
        'apex==0.9.10dev',
        'beautifulsoup4==4.11.1',
        'Bio==1.5.9',
        'biopython==1.81',
        'biotite==0.31.0',
        'einops==0.6.1',
        'einops_exts==0.0.4',
        'matplotlib==3.5.3',
        'nbconvert==7.6.0',
        'nbformat==5.9.0',
        'numpy==1.21.6',
        'openfold==0.0.1',
        'pandas==1.3.5',
        'pytest==7.4.0',
        'pytorchtools==0.0.2',
        'rdkit==2022.9.5',
        'requests==2.31.0',
        'scikit-learn==0.22.1',
        'scikit-learn==1.0.2',
        'scipy==1.7.3',
        'seaborn==0.12.2',
        'selenium==4.10.0',
        'setuptools==65.5.1',
        'statsmodels==0.13.2',
        'torch==1.12.0',
        'torch_geometric==2.2.0',
        'torch_scatter==2.1.0',
        'torch_sparse==0.6.15',
        'torchvision==0.13.0',
        'tqdm==4.64.1',
        'umap-learn==0.5.3',
        'xlrd==2.0.1',
        'xlutils==2.0.0',
        'xlwt==1.3.0',
    ],
)