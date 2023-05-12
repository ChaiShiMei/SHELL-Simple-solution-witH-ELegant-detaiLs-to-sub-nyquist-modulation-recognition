# SHELL solution for GBSense 2022 challenge
The repository contains the code for the solution of Team TII for the GBSense 2022 challenge

## Solution  
We develop a deep learning framework with elegant details for Sub-Nuqist modulation recognition. For the basic challenge, we porpose a variant ResNet fro 1-signal modulation recognition. For the advanced challenge, we decompose the challenge to 2 sub-task: spectrum sensing to detect the sub-band, and modulation recognition to identify the modulation pattern. A lightweight neural network and a modified ResNet are separately developed for these two tasks. Finally, we achieve the accuracy of 99.92% and 93.67% on the test dataset of both challenges.


## Installation
To run this reposity, first install conda from https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html and then create the environment by running the following commands:
```
conda env create -f environment.yml 
conda activate shell 
```

## Dataset and pretrained models:
Download the datasets and put them under the main folder'GBSense_TII'.

The pretrained models are stored in the folder 'Pretrained_model'. File 'best_checkpoint_basic.pt' is the best pretrained models of the basic challenge. Files 'best_checkpoint_ss.pt' and 'best_checkpoint_mr.pt' are the best pretrained models for spectrum sensing and modulation recognition of the advanced challenge.

The dataset and pretrained model are structured as follows:
```
- GBSense_TII
    - data_1
      -data_1_train.h5
      -data_1_test.h5
    - data_2
      -data_2_train.h5
      -data_2_test.h5
    -Pretrained_model
      -best_checkpoint_basic.pt
      -best_checkpoint_ss.pt
      -best_checkpoint_mr.pt
        ...
```


## Training and Evaluation
The details of training and testing codes are listed as follows:
* `Basic_Train.py`: training code of basic challenge.
* `Basic_Test.py`: test code for the unpublished test dataset of basic challenge.
* `Advanced_Spectrum_Sensing.py`: training code of spectrum sensing.
* `Advanced_Modulations_Recognition.py`: training code of modulation recognition.
* `Advanced_Test.py`: test code for the unpublished test dataset of advanced challenge.
For training, just run
```
python Basic_Train.py
python Advanced_Spectrum_Sensing.py
python Advanced_Modulations_Recognition.py
```
Hyperparameters (like bath size, learning rate, epoches) can be adjusted in the corresponding python files.
`Basic_Test.py` and `Advanced_Test.py` are designed for the unpulished test dataset. To run these two codes, just run
```
python Basic_Test.py --batch_size 256 --testset_root_path <path-to-unpublished-test-dataset>
```
and
```
python Advanced_Test.py --batch_size 256 --testset_root_path <path-to-unpublished-test-dataset> 
```
If meeting the 'torch.cuda.OutOfMemoryError', just reduce the batch_size.

The detected modulations for basic and advanced challenges will be separately saved in `BasicRecog.mat` and `AdvancedRecog.mat`. They have the same format with the targets 'Y' in the published dataset.


## Contact

*  Yu Tian, Kebin Wu
* { yu.tian, kebin.wu}@tii.ae

