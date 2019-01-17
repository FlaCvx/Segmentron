import argparse
import os
import nibabel as nib

import matplotlib.pyplot as plt

def show_liver_slice(index_slice, img_volume):
    slice = img_volume[index_slice, : , :]
    plt.figure()
    plt.imshow(slice, cmap="gray", origin="lower")

def show_differences(data_file, truth_file, prediction_file):
    count=0
    data = nib.load(os.path.abspath(data_file))
    truth = nib.load(os.path.abspath(truth_file))
    prediction = nib.load(os.path.abspath(prediction_file))

    fig, axeslist = plt.subplots(ncols=3, nrows=1)
    for slice1, slice2, slice3 in zip(data._data, truth._data, prediction._data):
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

    FLAGS, unparsed = parser.parse_known_args()

    prediction_dir = os.path.abspath("prediction")


    FLAGS.data_file=os.path.abspath(os.path.join(prediction_dir,FLAGS.data_file))
    FLAGS.prediction_file=os.path.abspath(os.path.join(prediction_dir,FLAGS.prediction_file))
    FLAGS.truth_file=os.path.abspath(os.path.join(prediction_dir,FLAGS.truth_file))
    show_differences(FLAGS.data_file, FLAGS.truth_file, FLAGS.prediction_file)



if __name__ == "__main__":
    main()
