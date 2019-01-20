import glob
import os
import warnings
import shutil

import SimpleITK as sitk
import dicom
import os
import numpy
from PIL import Image
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import argparse

#This config is modified starting from the one at brats.train
config = dict()

config["all_technologies"] = ["CT", "MR"]
config["training_technologie"] = ["CT"]  # change to "config["all_technologies"]" if you want to use all the technologies
config["training_modalities"] = [[ ],["T1DUAL","T2SPIR"]]  # change this if you want to only use some of the modalities. "T" indicates default


config["overwrite"] = False  # If True, will previous files. If False, will use previously written files.


def show_liver_slice(index_slice, img_volume):
    slice = img_volume[index_slice, : , :]
    plt.figure()
    plt.imshow(slice, cmap="gray", origin="lower")

def convert_image_format(in_file, out_file):
    sitk.WriteImage(sitk.ReadImage(in_file), out_file)
    return out_file

def correct_bias(in_file, out_file, image_type=sitk.sitkFloat64):
    """
    Corrects the bias using ANTs N4BiasFieldCorrection. If this fails, will then attempt to correct bias using SimpleITK
    :param in_file: input file path
    :param out_file: output file path
    :return: file path to the bias corrected image
    """

    tup = tuple(extract_dicom_paths(in_file))
    input_image = sitk.ReadImage(tup, image_type)
    output_image = sitk.N4BiasFieldCorrection(input_image, input_image > 0)
    #sitk.WriteImage(output_image, out_file) # In case you wanna save it as sitk image of whatever it is.

    new_arr = sitk.GetArrayFromImage(output_image)
    array_img = nib.Nifti1Image(new_arr, np.diag([1, 2, 3, 1]))
    nib.save(array_img, out_file)  # The file will be array_img.dataobj
    return os.path.abspath(out_file)

def extract_dicom_image(image_path, out_file, image_type=sitk.sitkFloat64):
    if "inPhase" in out_file: #Takes the images that have an even number, should be "in_phase"
        tup = tuple(extract_dicom_paths(image_path))
        input_image = sitk.ReadImage(tup, image_type)
        new_arr = sitk.GetArrayFromImage(input_image)
        #Take only even images
        new_arr = np.fliplr(np.array([new_arr[c,:,:] for c in range(0,len(new_arr),1) if c%2==0])) # Takes even images of the T1DUAL that should be
        # the ones associated with the in_phase modality.
        array_img = nib.Nifti1Image(new_arr, np.diag([1, 2, 3, 1]))
        nib.save(array_img, out_file)  # The file will be array_img.dataobj

    elif "outPhase" in out_file: #Takes the images that have an odd number, should be "out_phase"
        tup = tuple(extract_dicom_paths(image_path))
        input_image = sitk.ReadImage(tup, image_type)
        new_arr = sitk.GetArrayFromImage(input_image)
        # Take only even images
        new_arr = np.fliplr(np.array([new_arr[c, :, :] for c in range(0, len(new_arr), 1) if c % 2 != 0]))  # Takes even images of the T1DUAL that should be
        # the ones associated with the in_phase modality.
        array_img = nib.Nifti1Image(new_arr, np.diag([1, 2, 3, 1]))
        nib.save(array_img, out_file)  # The file will be array_img.dataobj

    else:

        tup = tuple(extract_dicom_paths(image_path))
        input_image = sitk.ReadImage(tup, image_type)
        new_arr = sitk.GetArrayFromImage(input_image)
        new_arr = np.fliplr(new_arr)
        array_img = nib.Nifti1Image(new_arr, np.diag([1, 2, 3, 1]))
        nib.save(array_img, out_file)  # The file will be array_img.dataobj



def get_image(subject_folder, name):
    file_card = os.path.join(subject_folder, "*" + name + ".nii.gz")
    try:
        return glob.glob(file_card)[0]
    except IndexError:
        raise RuntimeError("Could not find file matching {}".format(file_card))


def normalize_image(in_file, out_file, bias_correction=True):
    if bias_correction:
        correct_bias(in_file, out_file)
    else:
        shutil.copy(in_file, out_file)
    return out_file

def extract_ground_truth_image(image_path, out_file):
    lstFilesPng = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(image_path):
        for filename in fileList:
            if ".png" in filename.lower():  # check whether the file is PNG
                lstFilesPng.append(os.path.join(dirName, filename))

    lstFilesPng.sort()
    ref_pic = Image.open(lstFilesPng[0])
    dims = [len(lstFilesPng), ref_pic.size[0], ref_pic.size[1]]
    ArrayPng = numpy.zeros(dims)
    for i in range(0,len(lstFilesPng)):
        ArrayPng[i] =  np.array(Image.open(lstFilesPng[i]))

    new_arr=np.fliplr(ArrayPng)
    array_img = nib.Nifti1Image(new_arr, np.diag([1, 2, 3, 1]))
    if('CT' in out_file):
        new_arr[np.where(new_arr == 1)] = 80
    nib.save(array_img, out_file) #The file will be array_img.dataobj


def extract_dicom_paths(PathDicom):
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(PathDicom):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                lstFilesDCM.append(os.path.join(dirName, filename))

    # Get ref file
    RefDs = dicom.read_file(lstFilesDCM[0])
    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))

    # Load spacing values (in mm)
    ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

    x = numpy.arange(0.0, (ConstPixelDims[0] + 1) * ConstPixelSpacing[0], ConstPixelSpacing[0])
    y = numpy.arange(0.0, (ConstPixelDims[1] + 1) * ConstPixelSpacing[1], ConstPixelSpacing[1])
    z = numpy.arange(0.0, (ConstPixelDims[2] + 1) * ConstPixelSpacing[2], ConstPixelSpacing[2])

    # The array is sized based on 'ConstPixelDims'
    ArrayDicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

    # loop through all the DICOM files
    for filenameDCM in lstFilesDCM:
        # read the file
        ds = dicom.read_file(filenameDCM)
        # store the raw image data
        ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array

    lstFilesDCM.sort()
    return lstFilesDCM

def convert_livers_folder(in_folder, out_folder, bias_field_correction, truth_name='seg'):
    #The data given by the competition need to be processed beacuse there are different format files etc.
    # It converts everything to .nii (which hopefully will allow me to reuse code in the experiments.

    try:
        name_tec = ((in_folder.split("/")[len(in_folder.split("/"))-2]).split("_"))[0]
        if (name_tec=="CT"):
            image_file = os.path.abspath(os.path.join(in_folder,"DICOM_anon"))
            out_file = os.path.abspath(os.path.join(out_folder, name_tec + ".nii.gz"))
            if not bias_field_correction:
                extract_dicom_image(image_file, out_file, image_type=sitk.sitkFloat64) #Also saves it in out_file
            else:
                normalize_image(image_file, out_file )

            #Now extract the truth image.
            truth_file = os.path.abspath(os.path.join(in_folder, "Ground"))
            out_file = os.path.abspath(os.path.join(out_folder, name_tec + "_truth.nii.gz"))
            if not bias_field_correction:
                extract_ground_truth_image(truth_file, out_file)
            else:
                normalize_image(truth_file, out_file )


        elif (name_tec=="MR"):
            for name_mod in config["training_modalities"][1]:
                if name_mod is "T1DUAL":
                    #The T1 DUAL DATASET HAS THE "IN-PHASE" IMAGES & THE "OUT-PHASE" IMAGES. THEY BOTH
                    # HAVE THE SAME GROUND TRUTH IMAGE AS REFERENCE.
                    image_file = os.path.abspath(os.path.join(in_folder, name_mod, "DICOM_anon"))
                    in_phase_out_file = os.path.abspath(os.path.join(out_folder,name_tec+"_inPhase_"+name_mod+ ".nii.gz"))
                    out_phase_out_file = os.path.abspath(os.path.join(out_folder,name_tec+"_outPhase_"+name_mod+ ".nii.gz"))
                    if not bias_field_correction:
                        extract_dicom_image(image_file, in_phase_out_file, image_type=sitk.sitkFloat64)  # Also saves it in in_phase_out_file
                        extract_dicom_image(image_file, out_phase_out_file, image_type=sitk.sitkFloat64)  # Also saves it in out_phase_out_file
                    else:
                        normalize_image(image_file, out_file )

                else:
                    image_file = os.path.abspath(os.path.join(in_folder, name_mod, "DICOM_anon"))
                    out_file = os.path.abspath(os.path.join(out_folder,name_tec+"_"+name_mod+ ".nii.gz"))
                    if not bias_field_correction:
                        extract_dicom_image(image_file, out_file, image_type=sitk.sitkFloat64)  # Also saves it in out_file
                    else:
                        normalize_image(image_file, out_file )


                #Now extract the truth image.
                truth_file = os.path.abspath(os.path.join(in_folder, name_mod, "Ground"))
                out_file = os.path.abspath(os.path.join(out_folder, name_tec+"_"+name_mod+"_truth.nii.gz"))
                if not bias_field_correction:
                    extract_ground_truth_image(truth_file, out_file)  # Write in oout_file the extracted array from png images.
                else:
                    normalize_image(truth_file, out_file )

    except RuntimeError as error:
        raise error

def convert_livers_data(livers_folder, out_folder, bias_field_correction=False, overwrite=False ):
    """
    Preprocesses the LIVERS data and writes it to a given output folder. Assumes the original folder structure.
    :param livers_folder: folder containing the original livers data
    :param out_folder: output folder to which the preprocessed data will be written
    :param overwrite: set to True in order to redo all the preprocessing
    :return:
    """

    mr_ids = []
    for subject_folder in glob.glob(os.path.join(livers_folder, "*", "*")):
        if os.path.isdir(subject_folder):
            if('MR' in subject_folder):
                mr_ids.append(subject_folder.split("/")[3])


    for subject_folder in glob.glob(os.path.join(livers_folder, "*", "*")):
        if os.path.isdir(subject_folder): #Ignore everything that is not a directory, no problems for txt etc...
            subject = os.path.basename(subject_folder)
            new_subject_folder = os.path.join(out_folder, os.path.basename(os.path.dirname(subject_folder)),
                                              subject)
            if not os.path.exists(new_subject_folder) or overwrite:
                if not os.path.exists(new_subject_folder):
                    os.makedirs(new_subject_folder)
                convert_livers_folder(subject_folder, new_subject_folder, bias_field_correction, mr_ids )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Compress data to a fixed dimension')
    parser.add_argument('--original_data_dir', type=str, required='True', help='Directory with original data.', default='data/original')
    parser.add_argument('--output_data_dir', type=str, required='True', help='Directory where to put extracted data.', default='data/extracted')

    args = parser.parse_args()

    convert_livers_data(livers_folder=args.original_data_dir, out_folder=args.output_data_dir)

