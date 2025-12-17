# WLASL-Recognition-and-Translation

This repository contains the "WLASL Recognition and Translation", employing the `WLASL` dataset descriped in "Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison" by Dongxu Li.


>The project uses Cuda and pytorch, hence a system with NVIDIA graphics is required. Also, to run the system a minimum of 4-5 Gb of dedicated GPU Memory is needed.

### File Structure
-----------------

WLASL-Recognition-and-Translation/
├── README.md
└── WLASL/
    └── I3D/
        ├── __pycache__/
        │   ├── language.cpython-39.pyc
        │   ├── pytorch_i3d.cpython-39.pyc
        │   └── videotransforms.cpython-39.pyc
        ├── .gitattributes
        ├── archived/
        │   ├── asl100/
        │   │   ├── FINAL_nslt_100_iters=896_top1=65.89_top5=84.11_top10=89.92.ini
        │   │   └── FINAL_nslt_100_iters=896_top1=65.89_top5=84.11_top10=89.92.pt
        │   ├── asl1000/
        │   │   ├── FINAL_nslt_1000_iters=5104_top1=47.33_top5=76.44_top10=84.33.ini
        │   │   └── FINAL_nslt_1000_iters=5104_top1=47.33_top5=76.44_top10=84.33.pt
        │   ├── asl2000/
        │   │   ├── FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.ini
        │   │   └── FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt
        │   └── asl300/
        │       ├── FINAL_nslt_300_iters=2997_top1=56.14_top5=79.94_top10=86.98.ini
        │       └── FINAL_nslt_300_iters=2997_top1=56.14_top5=79.94_top10=86.98.pt
        ├── configfiles/
        │   ├── asl100.ini
        │   ├── asl1000.ini
        │   ├── asl2000.ini
        │   └── asl300.ini
        ├── configs.py
        ├── cpu-based-interface.py
        ├── cpu-based-interface2.0.py
        ├── cpu-based-reuirements.txt
        ├── cpu-based-run1.py
        ├── datasets/
        │   ├── __pycache__/
        │   │   ├── ms_wlasl_dataset.cpython-37.pyc
        │   │   ├── msasl_dataset.cpython-37.pyc
        │   │   ├── nslt_dataset_all.cpython-36.pyc
        │   │   ├── nslt_dataset_all.cpython-37.pyc
        │   │   ├── nslt_dataset_all.cpython-39.pyc
        │   │   ├── nslt_dataset.cpython-37.pyc
        │   │   └── siamese_dataset_train.cpython-37.pyc
        │   ├── nslt_dataset_all.py
        │   └── nslt_dataset.py
        ├── inference_i3d.py
        ├── json_extraction_N.py
        ├── language.py
        ├── models/
        │   ├── __pycache__/
        │   │   └── pytorch_i3d.cpython-37.pyc
        │   └── layers/
        │       └── __pycache__/
        │           └── SelfAttention.cpython-37.pyc
        ├── preprocess/
        │   ├── nslt_100.json
        │   ├── nslt_1000.json
        │   ├── nslt_2000.json
        │   ├── nslt_300.json
        │   ├── wlasl_2000_list.txt
        │   └── wlasl_class_list.txt
        ├── pytorch_i3d.py
        ├── requirements.txt
        ├── run.py
        ├── sentence_builder.py
        ├── test_i3d.py
        ├── test.py
        ├── train_i3d.py
        ├── videotransforms.py
        ├── webcam_interface.py
        └── weights/
            ├── flow_charades.pt
            ├── flow_imagenet.pt
            ├── rgb_charades.pt
            └── rgb_imagenet.pt


### Download Dataset
-----------------

The dataset used in this project is the "WLASL" dataset and it can be found [here](https://www.kaggle.com/datasets/utsavk02/wlasl-complete) on Kaggle

Download the dataset and place it in data/ (in the same path as WLASL directory)

### Steps to Run
-----------------

To run the project follow the steps

1. Clone the repo

 ```
 
 git clone https://github.com/alanjeremiah/WLASL-Recognition-and-Translation.git
 
 ```
 
2. Install the packages mentioned in the requirements.txt file


> Note: Need to install the correct compatible version of the cudatoolkit with pytorch. The compatible version with the command line can be found [here](https://pytorch.org/get-started/locally/). Below is the CLI used in this project


```

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

```

3. Open the WLASL/I3D folder and unzip the NLP folder in that path

4. Open the run.py file to run the application

```

python run.py

```

### Model
-----------------

This repo uses the I3D model. To train the model, view the original "WLASL" repo [here](https://github.com/dxli94/WLASL/blob/master/README.md)

### NLP
-----------------

The NLP models used in this project are the `KeyToText` and the `NGram` model. 

The KeyToText was built over T5 model by Gagan, the repo can be found [here](https://github.com/gagan3012/keytotext)

### Demo
-----------------

The end results of the project looks like this. 

The conversion of `Sign language` to Spoken Language.




https://user-images.githubusercontent.com/69314264/168775253-93a68a4c-8a22-4475-81f3-37393add6653.mp4





