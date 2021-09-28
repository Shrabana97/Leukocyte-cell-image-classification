# Leukocyte cell image classification

The proposed model is trained on NVIDIA RTX 2060 GPU paired with a quad-core processor \& 16gb of DDR4 memory clocked at 3000mHz.
\
\
The research paper is going to publised soon. Once the paper is published, the link of the paper will be shown here.

## Dataset source : 
* Acute Lymphoblastic Leukemia Image Database for Image Processing (ALL-IDB), Department of Computer Science, Università degli Studi di Milano, https://homes.di.unimi.it/scotti/all/

## Run in your own machine :
* `git clone https://github.com/Shrabana97/Leukocyte-cell-image-classification.git`
* `cd Leukocyte-cell-image-classification`
* `pip3 install -r requirements.txt` (install required libraries)
* To split the data run `python3 split.py` and follow the on-screen instructions.
    *   It will ask the address of the 'train' folder and will store that address in `path_of_train_dir.txt`. 
* To resize the data run `python3 pre-process.py`.
* To perform synthetic data augmentation:
    * Open `jupyter notebook`.
    * Run `DCGAN_benign.ipynb` to augment benign cell images.
    * Run `DCGAN_malignant.ipynb` to augment malignant cell images
* To train and create the model run `python3 train.py`.
    * Model and the training details will be stored in `result` directory.