import argparse
import os

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
        type=str,
        default=('80, 160, 240, 255'),
        required=True,
        help='Which labels to show. 80: Liver, 160: Right Kidney, 240: Left Kidney, 255:Spleen'
    )

    parser.add_argument(
        '--data_file',
        type=str,
        default='liver',
        required=True,
        help='String; The name for the data_file'
    )
    FLAGS, unparsed = parser.parse_known_args()

    prediction_dir = os.path.abspath("prediction")
    fixed_path="./Data_and_Pretrained_Models/Liver Region Segmentation/"
    more_fixed_path="./Data_and_Pretrained_Models/Liver Region Segmentation/Trained_models/"+"Labels_"+str(FLAGS.labels)+"/"

    prediction_dir = os.path.join(prediction_dir, FLAGS.model_file.split("/")[0])
    FLAGS.validation_file=os.path.abspath(fixed_path+str(FLAGS.validation_file))
    FLAGS.model_file=os.path.abspath(more_fixed_path+FLAGS.model_file)
    FLAGS.data_file=os.path.abspath(fixed_path+FLAGS.data_file)
    run_validation_cases(validation_keys_file=FLAGS.validation_file,
                         model_file=FLAGS.model_file,
                         training_modalities=FLAGS.training_technologies,
                         labels=FLAGS.labels,
                         hdf5_file=FLAGS.data_file,
                         output_label_map=True,
                         output_dir=prediction_dir)

if __name__ == "__main__":
    main()
