# Tensorflow 2 - Image classification

[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)
[![forthebadge](https://forthebadge.com/images/badges/powered-by-black-magic.svg)](https://forthebadge.com)

## About the project

These project aim to be an achievable introduction to machine learning image classification algorithm using Tensorflow 2 for beginners. It's based on the Tensorflow documentation and bring more details concerning the installation on your local machine.

_Please refer to the documentation for more details [Tensorflow](https://www.tensorflow.org/overview)_

&nbsp;

## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This project assumes that you're using the following configurations:

* Windows 10 or 11
* [WSL 2](https://learn.microsoft.com/fr-fr/windows/wsl/install)
* NVIDIA Graphic card

If you're not metting this requirements, please refer to the appropriate documentation according to your system configurations. Alternatively, if you have a AMD or Intel GPU you can use [DirectML](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-tensorflow-plugin).

PS: If you're a Mackbook user, maybe it's time to change your computer 😉.

### Installation

1. Check if your NVIDIA drivers are up to date. For that, you can use [Geforce Experience application](https://www.nvidia.com/fr-fr/geforce/geforce-experience/) or directly download them on the [NVIDIA website](https://www.nvidia.com/Download/index.aspx).
2. Check if WSL2 is up to date. If not, you can update it with the following command in a windows promt:

   ```sh
   wsl.exe --update
   ```

3. Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) on your Windows distribution. Please follow the instructions untill you've completed the installation.

4. Now, we need to install CUDA Toolkit on WSL2. First, open a WSL2 prompt and remove the old GPG key by enter the following command:

   ```bash
   sudo apt-key del 7fa2af80
   ```

   Then, you can process to the installation by entering the following commands:

   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
   ```

   ```bash
   sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
   ```

   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-wsl-ubuntu-12-1-local_12.1.0-1_amd64.deb
   ```

   ```bash
   sudo dpkg -i cuda-repo-wsl-ubuntu-12-1-local_12.1.0-1_amd64.deb
   ```

   ```bash
   sudo cp /var/cuda-repo-wsl-ubuntu-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
   ```

   ```bash
   sudo apt-get update
   ```

   ```bash
   sudo apt-get -y install cuda
   ```

   _Please refer to the official [documentation](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local) for more details concerning this step_

5. Install Miniconda to handle virtual environments in your WSL2 distribution by running the following commands:

   ```bash
   curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
   ```

   ```bash
   bash Miniconda3-latest-Linux-x86_64.sh
   ```

   Please follow the installation steps in the Miniconda prompt. After that you can restart your terminal and check the installed conda version by enter the following command:

   ```bash
   conda -V
   ```

   You should see something like:

   ```bash
   conda 23.3.0
   ```

6. Create a virtual env named tensorflow with python 3.9 by running the following command:

   ```bash
   conda create --name tensorflow python=3.9
   ```

7. If not activate, activate the tensorflow env by running the following command:

   ```bash
   conda activate tensorflow
   ```

8. Check the NVIDIA drivers installation:

   ```bash
   nvidia-smi
   ```

   You should see something like:

   ```bash
   Sat Apr  1 04:09:31 2023       
   +---------------------------------------------------------------------------------------+
   | NVIDIA-SMI 530.41.03              Driver Version: 531.41       CUDA Version: 12.1     |
   |-----------------------------------------+----------------------+----------------------+
   | GPU  Name                  Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
   | Fan  Temp  Perf            Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
   |                                         |                      |               MIG M. |
   |=========================================+======================+======================|
   |   0  NVIDIA GeForce RTX 3090         On | 00000000:2D:00.0  On |                  N/A |
   |  0%   28C    P8               33W / 420W|   2587MiB / 24576MiB |      4%      Default |
   |                                         |                      |                  N/A |
   +-----------------------------------------+----------------------+----------------------+
                                                                                            
   +---------------------------------------------------------------------------------------+
   | Processes:                                                                            |
   |  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
   |        ID   ID                                                             Usage      |
   |=======================================================================================|
   |  No running processes found                                                           |
   +---------------------------------------------------------------------------------------+
   ```

9. Now you need to install CUDA toolkit and some dependencies in the conda environment:

   ```bash
   conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
   ```

10. Create the config repository access for conda:

    ```bash
    mkdir -p $CONDA_PREFIX/etc/conda/activate.d
    ```

11. Export the repository path access for conda:

    ```bash
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    ```

    Restart your terminal and reactivate your conda env before passing to the next step.

12. Now, still in your conda tensorflow env, install tensorflow by running the following commands:

    ```bash
    python3.9 -m pip install --upgrade pip
    ```

    ```bash
    python3.9 -m pip install tensorflow==2.8.0
    ```

    You can test if the installation was success by running:

    ```bash
    python3.9 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
    ```

&nbsp;

## Usage

### Trainning

   ```bash
    python3.9 model_training.py
   ```

### Inference

   ```bash
    python3.9 python3.9 model_inference.py
   ```

### Tips

* [mnist_model.h5](mnist_model.h5) is the train model to format HDF5 format. You can export/import it where ever you want !
* See [exomodel.py](exomodel.py) for more details about loading pre-train model
* [hypertunning_logs](hypertunning_logs) directory is generated ONLY at the first train.
* If you want to reset the hyperparameters search, you just need to remove [hypertunning_logs](hypertunning_logs) directory and restart training
* Feel free to re-adjust the hypermodel as you whish

&nbsp;

## Acknowledgments

Resources you may find helpful !

* [Python 3.9 documentation](https://docs.python.org/3.9/)
* [Tensorflow 2.12.0 documentation](https://www.tensorflow.org/api_docs/python/tf)
* [NVIDIA drivers](https://www.nvidia.com/Download/index.aspx)
* [WSL2 installation](https://learn.microsoft.com/fr-fr/windows/wsl/install)
* [CUDA Toolkit Windows](https://developer.nvidia.com/cuda-downloads)
* [CUDA Toolkit GNU/Linux](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)
* [Miniconda GNU/Linux repository](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)
