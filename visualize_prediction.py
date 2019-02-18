import argparse
import os
import nibabel as nib
import pandas as pd
import collections
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist


def show_liver_slice(index_slice, img_volume):
    slice = img_volume[index_slice, : , :]
    plt.figure()
    plt.imshow(slice, cmap="gray", origin="lower")



def show_differences(data, truth, prediction, labels):
    count=0

    new_truth = np.copy(truth)
    new_truth[:,:,:] = 0
    for label in labels:
        label=int(label)
        a = np.where( truth.__array__() == label )
        new_indices = [(tuple1, tuple2, tuple3) for tuple1, tuple2, tuple3 in zip(a[0], a[1], a[2])]
        for ind in new_indices:
            new_truth[ind[0], ind[1], ind[2]] = label

    # Plot sequence of slices of livers
    fig, axeslist = plt.subplots(ncols=3, nrows=1)

    for slice1, slice2, slice3 in zip(data, new_truth, prediction):
        count += 1
        fig.suptitle('Slice: '+str(count), fontsize=20)
        axeslist.ravel()[0].imshow(slice1, cmap=plt.gray())
        axeslist.ravel()[0].set_title("Given data")
        axeslist.ravel()[0].set_axis_off()

        axeslist.ravel()[1].imshow(slice2, cmap=plt.gray())
        axeslist.ravel()[1].set_title("Ground truth")
        axeslist.ravel()[1].set_axis_off()

        axeslist.ravel()[2].imshow(slice3, cmap=plt.gray())
        axeslist.ravel()[2].set_title("Prediction")
        axeslist.ravel()[2].set_axis_off()

        plt.pause(.1)
        plt.draw()
        plt.tight_layout

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_file',
        type=str,
        default=None,
        required=True,
        help='Name of the data file to be used'
    )

    parser.add_argument(
        '--prediction_file',
        type=str,
        default=None,
        required=True,
        help='Name of the prediction file to be used'
    )

    parser.add_argument(
        '--truth_file',
        type=str,
        default=None,
        required=True,
        help='Name of the truth file to be used'
    )

    parser.add_argument(
        '--modality',
        type=str,
        default=None,
        required=True,
        help='Name modality used in the truth file'
    )

    parser.add_argument(
        '--labels',
        type=str,
        default=('80, 160, 240, 255'),
        help='Which labels to detect. 80: Liver, 160: Right Kidney, 240: Left Kidney, 255:Spleen'
    )


    FLAGS, unparsed = parser.parse_known_args()

    f1 = lambda s: '_'.join([(item) for item in s.split(',')])

    prediction_dir = os.path.join(os.path.abspath("prediction"),"Labels_"+f1(FLAGS.labels.replace(" ","")))

    FLAGS.data_file=os.path.abspath(os.path.join(prediction_dir,FLAGS.data_file))
    FLAGS.prediction_file=os.path.abspath(os.path.join(prediction_dir,FLAGS.prediction_file))
    FLAGS.truth_file=os.path.abspath(os.path.join(prediction_dir,FLAGS.truth_file))

    data = nib.load(os.path.abspath(FLAGS.data_file))
    truth = nib.load(os.path.abspath(FLAGS.truth_file))
    prediction = nib.load(os.path.abspath(FLAGS.prediction_file))


    show_differences(data._data.__array__(), truth._data.__array__(), prediction._data.__array__(), FLAGS.labels.replace(" ","").split(","))



if __name__ == "__main__":
    main()




