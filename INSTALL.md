sometimes, hyak removes the files related to environments because we use scratch storage. So It may be required that we install an environment frequently:

1. Run the following command editing the directory location accordingly:
```bash
conda create -p saxspy "python>=3.8" -y
```
Then activate the environment using `conda activate saxspy` if the environments location is already added to the conda directory:

2. Then we need to install few packages. Start with gpytorch because this installs majority of packages used:
```bash
pip install gpytorch
```

3. Install matplotlib and seaborn.
```bash
pip install seaborn
```

4. neural-process packes requires the following additional packages
```bash
pip install torchvision
```