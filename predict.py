import argparse
import os
import nibabel as nib
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

from scipy.spatial import cKDTree

from Unet3D.unet3d.prediction import run_validation_cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--validation_file',
        type=str,
        default=None,
        required=True,
        help='Name of the validation file to be used'
    )

    parser.add_argument(
        '--model_file',
        type=str,
        default=None,
        required=True,
        help='Name of the model file to be used'
    )

    parser.add_argument(
        '--training_technologies',
        type=lambda s: tuple([str(item) for item in s.split(',')]),
        default=('CT', 'MR'),
        required=True,
        help='names of the training technologies files.'
    )

    parser.add_argument(
        '--labels',
        type=lambda s: tuple([int(item) for item in s.split(',')]),
        default=(80, 160, 240, 255),
        required=False,
        help='Which labels to show. 80: Liver, 160: Right Kidney, 240: Left Kidney, 255:Spleen'
    )

    parser.add_argument(
        '--data_file',
        type=str,
        default='liver',
        required=True,
        help='String; The name for the data_file'
    )

    parser.add_argument(
        '--interpolation',
        type=str,
        default='nearest',
        required=False,
        help='Interpolation to be used to restore original shape. '
    )
    FLAGS, unparsed = parser.parse_known_args()

    prediction_dir = os.path.abspath("prediction")
    fixed_path="./Data_and_Pretrained_Models/Liver Region Segmentation/"

    f1 = lambda s: '_'.join([str(item) for item in s])
    more_fixed_path=os.path.join(".","Data_and_Pretrained_Models","Liver Region Segmentation","Trained_models","Labels_"+f1(FLAGS.labels)+"/")

    prediction_dir = os.path.join(prediction_dir,"Labels_"+f1(FLAGS.labels), FLAGS.model_file.split("/")[0])

    FLAGS.validation_file=os.path.abspath(fixed_path+str(FLAGS.validation_file))
    FLAGS.model_file=os.path.abspath(more_fixed_path+FLAGS.model_file)
    FLAGS.data_file=os.path.abspath(fixed_path+FLAGS.data_file)
    if not (os.path.exists(prediction_dir)):
        run_validation_cases(validation_keys_file=FLAGS.validation_file,
                             model_file=FLAGS.model_file,
                             training_modalities=FLAGS.training_technologies,
                             labels=FLAGS.labels,
                             hdf5_file=FLAGS.data_file,
                             interpolation=FLAGS.interpolation,
                             output_label_map=True,
                             output_dir=prediction_dir)
        print("Prediction saved in: "+prediction_dir)



    print("Metrics: ")

    prediction_dir = (glob.glob(os.path.join( prediction_dir, "*"))[0])
    truth = nib.load(os.path.abspath(prediction_dir)+"/truth.nii.gz")
    prediction = nib.load(os.path.abspath(prediction_dir)+"/prediction.nii.gz")

    CT_modality = FLAGS.data_file.find("CT")
    if (CT_modality > -1):
        modality='CT'
    else:
        modality='MR'
    show_metrics_table( truth._data, prediction._data, FLAGS.labels, modality=modality)




def show_metrics_table(truth_file, prediction_file, labels, modality):

    truth_file=truth_file.__array__()
    prediction_file = prediction_file.__array__()
    columns = ['Volumetric overlap', 'Relative absolute volume difference', 'Average symmetric surface distance',
               'Root mean square symmetric surface distance', 'Maximum symmetric surface distance']

    ASSD = None
    RMSD = None
    MSSD = None
    d = {}

    if (modality=='CT'):
        print("Modality is 'CT', therefore print only liver metrics")
        truth_file[np.where(truth_file != 80)] = 0  # Set to zero all the elements that do not have label==80
        truth_file[np.where(truth_file == 80)] = 1

        prediction_file[np.where(prediction_file != 80)] = 0
        prediction_file[np.where(prediction_file == 80)] = 1
        VO, RAVD = volumetric_overlap_error(truth_file, prediction_file)
        #ASSD, RMSD, MSSD = Average_symmetric_absolute_surface_distance(truth_file, prediction_file)

        d['Liver'] =  [ VO, RAVD, ASSD, RMSD, MSSD  ]
    else:
        print("Print the liver metrics of the following labels: ", labels)
        copy_truth = np.copy(truth_file)
        copy_prediction = np.copy(prediction_file)
        for label in labels:
            truth_file=np.copy(copy_truth)   #If I have more than one label, each time I set to zero all the other labels.
            prediction_file=np.copy(copy_prediction)
            if label==80:
                truth_file[np.where(truth_file != 80)] = 0  #Set to zero all the elements that do not have label==80
                truth_file[np.where(truth_file==80)] = 1    #Take all the elements that have label==80.

                prediction_file[np.where(prediction_file != 80)] = 0
                prediction_file[np.where(prediction_file==80)] = 1
                VO, RAVD = volumetric_overlap_error(truth_file, prediction_file)
                #ASSD, RMSD, MSSD = Average_symmetric_absolute_surface_distance(truth_file, prediction_file)
                d['Liver'] = [VO, RAVD, ASSD, RMSD, MSSD ]

            if label==160:
                truth_file[np.where(truth_file != 160)] = 0
                truth_file[np.where(truth_file == 160)] = 1

                prediction_file[np.where(prediction_file != 160)] = 0
                prediction_file[np.where(prediction_file==160)] = 1
                VO, RAVD = volumetric_overlap_error(truth_file, prediction_file)
                #ASSD, RMSD, MSSD = Average_symmetric_absolute_surface_distance(truth_file, prediction_file)

                d['Right Kidney'] = [VO, RAVD, ASSD, RMSD, MSSD ]

            if label==240:

                truth_file[np.where(truth_file != 240)] = 0
                truth_file[np.where(truth_file == 240)] = 1

                prediction_file[np.where(prediction_file != 240)] = 0
                prediction_file[np.where(prediction_file==240)] = 1
                VO, RAVD = volumetric_overlap_error(truth_file, prediction_file)
                #ASSD, RMSD, MSSD = Average_symmetric_absolute_surface_distance(truth_file, prediction_file)
                d['Left Kidney'] = [VO, RAVD, ASSD, RMSD, MSSD ]
            if label==255:

                truth_file[np.where(truth_file !=255)] = 0
                truth_file[np.where(truth_file == 255)] = 1

                prediction_file[np.where(prediction_file != 255)] = 0
                prediction_file[np.where(prediction_file==255)] = 1
                VO, RAVD = volumetric_overlap_error(truth_file, prediction_file)
                #ASSD, RMSD, MSSD = Average_symmetric_absolute_surface_distance(truth_file, prediction_file)
                d['Spleen'] = [ VO, RAVD, ASSD, RMSD, MSSD ]

    df = pd.DataFrame(data=d)
    print(df)


def volumetric_overlap_error( truth, prediction):

    # Volumetric overlap error. This is the number of voxels in the
    # intersection of segmentation and reference divided by the number of voxels
    # in the union of segmentation and reference. This value is 1 for a perfect
    # segmentation and has 0 as the lowest possible value, when there is no overlap
    # at all between segmentation and reference.

    overlap = 0
    truth_vol = 0
    prediction_vol= 0
    union_vol = 0
    for truth_slice, prediction_slice in zip(truth, prediction):
        new_overlap, new_truth_vol, new_prediction_vol, new_union = count_slice_overlap(truth_slice, prediction_slice)
        overlap += new_overlap
        truth_vol += new_truth_vol
        prediction_vol += new_prediction_vol
        union_vol += new_union

    if(union_vol!=0):
        VO = overlap/union_vol
    else:
        VO=0
    # Relative absolute volume difference(RAVD): Also provides information about the differences
    # between volumes between segmented and reference organs, but values the differences more
    # than overlap(0 for a perfect segmentation).
    #
    RAVD = abs(prediction_vol-truth_vol)
    return VO, int(RAVD)


def count_slice_overlap(truth_slice, prediction_slice):


    truth_indices = np.where(truth_slice==1) #indexes where there is the correct segmentation
    prediction_indices = np.where(prediction_slice==1) #indexes that I predicted to be the corrected segmentation

    truth_indices = np.stack((truth_indices[0], truth_indices[1]), axis=-1)
    prediction_indices = np.stack((prediction_indices[0], prediction_indices[1]), axis=-1)

    overlap = len(np.intersect1d(truth_indices,prediction_indices))
    area_truth=len(truth_indices)
    area_prediction=len(prediction_indices)

    union = area_truth+area_prediction-overlap #the overlap is coutned twice.
    return overlap, area_truth, area_prediction, union


def Average_symmetric_absolute_surface_distance(truth_vol, prediction_vol):
    # Average symmetric absolute surface distance, in millimeters.
    # The border voxels of segmentation and reference are determined.
    # These are defined as those voxels in the object that have at
    # least one neighbour (from the 26 nearest neighbours) that does
    # not belong to the object. For each voxel in these sets, the closest
    # voxel in the other set is determined (using Euclidean distance and
    # real world distances, so taking into account the generally different
    # resolutions in the different scan directions). All these distances
    # are stored, for border voxels from both reference and segmentation.
    # The average of all these distances gives the averages symmetric absolute
    # surface distance. This value is 0 for a perfect segmentation.

    indices_truth = np.where(truth_vol==1)
    indices_prediction = np.where(prediction_vol == 1)

    i=0
    new_indices_truth = []
    new_indices_prediction = []
    for i in range(0,len(indices_truth[0])):
        new_indices_truth.append([indices_truth[0][i], indices_truth[1][i], indices_truth[2][i]])

    for i in range(0, len(indices_prediction[0])):
        new_indices_prediction.append([indices_prediction[0][i], indices_prediction[1][i], indices_prediction[2][i]])

    indices_truth=new_indices_truth
    indices_prediction = new_indices_prediction

    truth_border = []
    for index in indices_truth:
        x = index[0]
        y = index[1]
        z = index[2]
        if ((len(np.where(truth_vol[x-1:x+2,y-1:y+2,z-1:z+2]== 0))) > 0):
            truth_border.append([x,y,z])

    prediction_border = []
    for index in indices_prediction:
        x = index[0]
        y = index[1]
        z = index[2]
        if ((len(np.where(prediction_vol[x-1:x+2,y-1:y+2,z-1:z+2]== 0))) > 0):
            prediction_border.append([x,y,z])


    #Now truth_border and prediction_border are the lists of indices of the border voxels.
    #Now find the

    min_dists, min_dist_idx = cKDTree(truth_border).query(prediction_border, 1)
    Y = cdist(truth_border, prediction_border, 'euclidean')
    minimum_distances_truth = Y.min(axis=1)
    minimum_distances_prediction = Y.min(axis=0)

    in_first = set(minimum_distances_truth)
    in_second = set(minimum_distances_prediction)

    in_second_but_not_in_first = in_second - in_first

    minimum_distances = list(minimum_distances_truth) + list(in_second_but_not_in_first)
    ASSD = np.mean(minimum_distances)

    # Symmetric RMS surface distance, in millimeters.This measures is similar to the previous measure, but
    # stores the squared distances between the two sets of border voxels.After averaging the squared values, the
    # root is extracted and gives the symmetric RMS surface distance.This value is 0 for a perfect segmentation.

    squared_distance = []
    for element in minimum_distances:
        squared_distance.append(element*element)
    RMSD = np.sqrt(np.mean(ASSD))

    # Maximum symmetric absolute surface distance, in millimeters.This measure is similar to the previous two, but only
    # the maximum of all voxel distances is taken instead of the average.This value is 0 for a perfect segmentation

    MSSD = np.max(minimum_distances)
    return ASSD, RMSD, MSSD



if __name__ == "__main__":
    main()
