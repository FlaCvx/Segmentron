# Segmentron
### Life-long learning approach to Medical image segmentation of Liver and other abdominal organs. Based on the online competition: https://chaos.grand-challenge.org/

Getting started
===========

### Prerequisites:
#### Installation

1- Create a virtual environment with python 3.6 and activate it.

2- Install the depencies listed in "requirements.txt".

```
cd Segmentron

pip install -r requirements.txt

```

### Data processing:
First, you need to download the data at: https://chaos.grand-challenge.org/Download/.
Unzip them and put the unzipped files "CT_data_batch1" and "MR_data_batch1" in a directory called "data/original".
Then, place "data" inside "Segmentron".

The livers data of the competition are saved as dicom images therefore we first need to collect them under a .nii.gz file
which will then be used in the training.


#### 1. EXTRACTION OF THE IMAGES:

The extraction of the dataset is done with the "extract_data.py" as follow:

```
cd Segmentron

python extract_data.py --original_data_dir "data/original" --output_data_dir "data/extracted"

```

The dataset consist of data created from two technologies : "CT" and "MR". The MR has two subdatasets, one "in-phase" and the other "out-phase" both take the same images as ground truth .
The images from CT and MR are given as .dicom files, while the ground truth images are given as PNGs' , all these files were read and transformed in .nii.gz files.
After the above mentioned step, the extracted dataset, located in &output_data_dir will have two subdirectories: "CT_data_batch1" nd "MR_data_batch1", both will have a number of subdirectories equal to the initial number of patients.
As for the CT's patient subdirectory, there will be two files "CT.nii.gz" and  "CT_truth.nii.gz", respectively corresponding to the data and the ground truth files.
While, for the MR's patient subdirectory, there are 5 files: "MR_inPhase_T1DUAL.nii.gz", "MR_outPhase_T1DUAL.nii.gz" with the corresponding ground truth file in "MR_T1DUAL_truth.nii.gz" (same for inPhase and outPhase data) and "MR_T2SPIR.nii.gz" with the ground truth in "MR_T2SPIR_truth.nii.gz" 

#### 2. TRAINING:
Once the extracted data are ready, you can procede by executing the "Parsing.py" file, an example is shown below.
Read the arguments in the main of "Parsing.py" to have a more clear understanding of the input parameters.

```
cd Segmentron

python Parsing.py --data_file "liver" --Base_directory "Liver Region Segmentation" --num_GPU 2 --data_directory "./data/extracted_small"

```


How to Cite
===========
If you use this code or the pretrained models in your research,
please cite:


Reference(s)
===========
Check here for more details: https://github.com/ellisdg/3DUnetCNN
