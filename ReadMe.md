# Sensitivity analysis of activations with augmented inputs

This repository contains source codes and Jupyter notebooks developed under the following study:
- *P.Kharyuk, S.Matveev, I.Oseledets.* **Exploring specialization and sensitivity of convolutional neural networks in the context of simultaneous augmentations.**

Complementary computational results are available at the corresponding Zenodo repository:
- https://doi.org/10.5281/zenodo.10499818

## Installation

The developed scripts and source codes are intended to be run in Unix systems supporting conda package manager and all necessary dependencies. 

Install ```conda``` package manager:
- https://docs.conda.io/projects/miniconda/en/latest/

Install all dependencies described in the ```environment.yml``` file (includes version numbers) and create the corresponding environment (```activation_sense```):
- ```conda env create -f environment.yml```

Activate created environment:
- ```conda activate activation_sense```

Enter the repository's directory, create a new session in ```tmux```, ```screen``` or other terminal manager, and start the Jupyter lab server:
- ```jupyter lab --port=8889 --no-browser```

The last option should be used if the notebook is running on the remote cluster and to be accessed remotely via ssh connection. In this case use the following command on your computer to establish ssh bridge (replace username and cluser_address with yours):
- ```ssh -N -f -L localhost:8889:localhost:8888 username@cluster_address```

Then enter a https://localhost:8888/ line in your browser and provide the ```token``` line (can be found within output of ```jupyter lab list```) to connect to the Jupyter lab server.

To transfer large files between laptop and cluster, consider using the ```rsync``` utility:
- ```rsync -LvzP username@cluster_address:path_to_file local_path_to_file```

## Preparing data and models

The ILSVRC dataset used in this work may be downloaded from the corresponding competition page at Kaggle:
https://www.kaggle.com/competitions/imagenet-object-localization-challenge

We provide the downloading and tuning scripts as a part of this repository. Please use the ```1-1_Download_ILSVRC_imagenet_data.ipynb``` (Kaggle account's username and key are required).

Also this notebook allows one to download the pretrained models used in this study:
- ```alexnet-owt-4df8aa71.pth``` (244.4 MB)
- ```vgg11-bbd30ac9.pth``` (531.5 MB)
- ```resnet18-5c106cde.pth``` (46.8 MB)


## Computing

All computations were performed on the local cluster with the following specifications:
- Hardware: 2x Intel(R) Xeon(R) CPU E5-2670 v3 @ 2.30GHz; 500 GB RAM; 1x GeForce GTX 1080 Ti GPU, 2x Tesla K80 GPU.
- OS: Linux Mint 17.3 Rosa (GNU/Linux 4.4.0-112-generic x86_64)
- NVIDIA Driver Version: 430.64, CUDA Version: 10.1

Basic computations involve using ```sh```-scripts encapsulating all necessary preparations and parameters. To generate these ```sh``` scripts, one should run the corresponding cells of the following Jupyter notebooks:
- Experimental series 1 (sensitivity values): ```2-1_Alexnet_sensitivity_analysis.ipynb```, ```2-2_VGG11_sensitivity_analysis.ipynb```, ```2-3_Resnet18_sensitivity_analysis.ipynb```;
- Experimental series 2 (guided masking predictions): ```7_Masked_predictions_analysis.ipynb```;
- Experimental series 3 (single channelled segments): ```4-1_HSV_single_channel_single_unit_Alexnet.ipynb```, ```4-2_HSV_single_channel_single_unit_Resnet18.ipynb``` .

**Note:** with the original number of samples, the computations of the 1st and 3rd experimental series require a huge disk space amount (up to 10TB approximately, for keeping intermediate results) and last for a long time (see S7.9 in the Supplementary S7). We organized these computations in a serial way, starting 1-2 scripts simultaneosly and removing intermediate computations (sampled activations) from the ```results``` directory before moving to the next part.

## Jupyter notebooks

All experiments were systematized using Jupyter Notebooks. We provide rendered versions of them as a part of the current repository. To reproduce the output, please use computational results provided within the abovementioned Zenodo repository as a reference (or recompute them manually, see the **note** in the **Computing section**). These files were additionally repacked after computing in order to reduce their size.

Below we describe the notebooks relating them to the corresponding experiments:

- Preparation:
    - ```1-1_Download_ILSVRC_imagenet_data.ipynb```: download data and pretrained models;
    - ```1-2_Display_ILSVRC_imagenet_data.ipynb```: observe data and augentations considered in the study;
- Experimental series 1 (sensitivity values):
    - ```2-1_Alexnet_sensitivity_analysis.ipynb```, ```2-2_VGG11_sensitivity_analysis.ipynb```, ```2-3_Resnet18_sensitivity_analysis.ipynb```: configure experiments, generate ```sh``` files and plot the sensitivity values for AlexNet, VGG11, ResNet18;
    - ```3-1_Alexnet_single_unit_sensitivity_analysis.ipynb```, ```3-2_VGG11_single_unit_sensitivity_analysis.ipynb```, ```3-3_Resnet18_single_unit_sensitivity_analysis.ipynb```: plot these sensitivity values for every single unit separately;
    - ```5-1_Correlation_analysis.ipynb```: checkpoint-wise correlation analysis of the SA variables;
    - ```6_Discriminant_analysis.ipynb```: convolutional checkpoint-wise Linear Discriminant analysis of the SA variables using their maps as features;
- Experimental series 2 (guided masking predictions):
    - ```7_Masked_predictions_analysis.ipynb```: configure experiments, generate ```sh``` files and display the results of guided masked predictions for AlexNet, VGG11, ResNet18;
    - ```8_Single_class_prediction_analysis.ipynb```: analyze the sensitivity of the last classifying layers and relate it to the sensitivity of the whole network;
- Experimental series 3 (single channelled segments):
    - ```4-1_HSV_single_channel_single_unit_Alexnet.ipynb```, ```4-2_HSV_single_channel_single_unit_Resnet18.ipynb```: configure experiments, generate ```sh``` files and plot the sensitivity values of selected units for AlexNet, VGG11, ResNet18 (converting input to HSV, splitting by single channels, muliplexing them for the respective segments);
    - ```5-2_HSV_single_channel_correlation_analysis.ipynb```: segment-wise correlation analysis of the SA variables for the standalone segments;
    - ```alexnet_single_units.json```, ```resnet18_single_units.json```: these files define the parts of the networks to be analyzed in the 3rd experimental series.

## Test cases

In order to check the core functionality, separate test cases are provided. These cases are ```sh``` scripts for computing activations, sensitivity values, masked predictions for a single network (Alexnet) with a limited number of checkpoints, selected units and masking approaches. These scripts can be found in the ```test``` directory.

Along with the scripts, we provide the reference files produced by a local machine (packed as ```4_test_references.7z``` within the Zenodo repository). Corresponding computations were performed on the laptop with the following configuration:
- Hardware: 1x Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz; 16 GB RAM; 1x GeForce MX150 GPU
- OS: Linux Ubuntu 22.04 (GNU/Linux 6.2.0-060200-generic x86_64)
- NVIDIA Driver Version: 535.54.03, CUDA Version: 12.2

Several details regarding these files are listed in ```test/Stats.md``` file.


## Repository structure

- ```data```: directory for storing dataset (will be created before downloading);
- ```experiments```: Basic ```sh```-files for performing computational experiments;
- ```notebook```: Jupyter notebooks storing the experimental setup and processing the results;
- ```results```: Directory for storing computed results (will be created before computing);
    - ```results/single_unit```: A separate directory for single-channelled experiments;
    - ```results/test```: A directory to store results related to testing (test);
    - ```results/tmp```: This directory is used to keep intermediate computations before merging them into single files;
- ```src```: ```py```-files containing the functions developed to maintain the experiments;
    - ```src/correlation```: utilities for performing correlation analysis;
    - ```src/data_loader```: loading ILSVRC images as a torchvision dataset;
    - ```src/discriminant_analysis```: linear discriminant analysis for predicting SA variables by their sensitivity maps;
    - ```src/image_transforms```: image augmentations considered in the study;
    - ```src/models```: loading pre-trained models;
    - ```src/prediction```: single-class sensitivity analysis and guided masking prediction sources (including correlations and HCA);
    - ```src/preparation```: single-channelled (HSV separated-and-multiplexed) computations;
    - ```src/sensitivity_analysis```: building sensitivity analysis problems, and computing the corresponding sensitivity values;
- ```templates```: Templates for forming ```sh```-scripts from Jupyter notebooks to be used for submitting computational tasks;
- ```test```: ```sh```-files configured to test computational core of the framework;
- ```torch-models```: directory for storing pretrained models available in torch (will be created before downloading).
