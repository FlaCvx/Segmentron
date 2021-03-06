import tensorflow as tf

from Unet3D.Adaptive import train_isensee2017_EWC as train_isensee2017
import gpustat
import time
import os
import argparse
import numpy as np
import pdb
from tensorflow.python.client import device_lib

# def get_free_gpus(num, except_gpu=None):
#     gpu_list = gpustat.GPUStatCollection.new_query()
#     free_gpu = []
#     while not len(free_gpu) == num:
#         for gpu in gpu_list:
#             if not gpu.processes:
#                 if not gpu.index == except_gpu:
#                     free_gpu.append(gpu.index)
#             if len(free_gpu) == num:
#                 break
#         if not len(free_gpu) == num:
#             free_gpu = []
#             print('Not enough GPUs avaialble at this time. Waiting ....')
#             time.sleep(20)
#     return free_gpu

def get_free_gpus():
    local_device_protos = device_lib.list_local_devices()

    return [x.name.split(":")[2] for x in local_device_protos if x.device_type == 'GPU']


def main():

    #FLAGS.GPU = get_free_gpus(FLAGS.num_GPU)
    FLAGS.GPU = get_free_gpus()
    print("Free gpus: ", FLAGS.GPU)
    train_isensee2017.config.update(vars(FLAGS))
    model = train_isensee2017.main(overwrite=False)
    model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_GPU',
        type=int,
        default=2,
        help='Integer; defines how many GPUs to use'
    )
    parser.add_argument(
        '--Base_directory',
        type=str,
        default='Liver Region Segmentation',
        help='String; New Folder where The Model will be stored.'
    )
    parser.add_argument(
        '--logging_file',
        type=str,
        default='None',
        help='String; names the file that stores the training progress'
    )
    parser.add_argument(
        '--image_shape',
        type=lambda s: tuple([int(item) for item in s.split(',')]),
        default=(128, 512, 512), #Max dimension of liver images. #I put the last to 512 because it makesthe train_isensee work
        #TODO: Check with luca that I can do it. Altrimenti la linea 111 di isensee2017_GPU_EWC non funziona...
        help='Tuple; The dimensions of the max input image, e.g. : 512, 512, 123'
    )
    parser.add_argument(
        '--normalize',
        type=str,
        default='hist_norm',
        help='options: mean_and_std, No, zero_one, hist_norm'
    )
    parser.add_argument(
        '--deconvolution',
        type=bool,
        default=True,
        help='Bool; whether to use convolution or upsampling'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Int; batch size'
    )
    parser.add_argument(
        '--transfer_model_file',
        type=str,
        default=None,
        help='String; The transfer model to start training with'
    )
    parser.add_argument(
        '--model_file',
        type = str,
        default='None',
        help='String; The name for the new model'
    )
    parser.add_argument(
        '--data_file',
        type=str,
        default='Data',
        help='String; The name for the data_file'
    )
    parser.add_argument(
        '--data_directory',
        type=str,
        default='None',
        help='String; The name for the data_file'
    )
    parser.add_argument(
        '--training_file',
        type=str,
        default='None',
        help='String; The name for the new model'
    )
    parser.add_argument(
        '--validation_file',
        type=str,
        default='None',
        help='String; The name for the new model'
    )
    parser.add_argument(
        '--flip',
        type=bool,
        default=False,
        help='whether to randomly flip axis during training'
    )
    parser.add_argument(
        '--distort',
        type=bool,
        default=False,
        help='whether to randomly flip axis during training'
    )
    parser.add_argument(
        '--return_name_only',
        type=bool,
        default=False,
        help='Return name of the Model only'
    )
    parser.add_argument(
        '--Except_layers',
        type=lambda s: tuple([str(item) for item in s.split(',')]),
        default=('conv3d_30', 'conv3d_22', 'conv3d_26'),
        help='Return name of the Model only'
    )
    parser.add_argument(
        '--ID',
        type=str,
        default='exp_2',
        help='In case the same config is used twice'
    )
    parser.add_argument(
        '--train_decoder_only',
        type=bool,
        default=False,
        help='Whether to train the decoder only'
    )
    parser.add_argument(
        '--Load_optimizer',
        type=bool,
        default=False,
        help='Whether to load the optimizer weights too'
    )
    parser.add_argument(
        '--fisher_multiplier',
        type=float,
        default=1000,
        help='How to scale the Fisher Info'
    )
    parser.add_argument(
        '--labels',
        type=str,
        default='80, 160, 240, 255',
        help='Which labels to detect. 80: Liver, 160: Right Kidney, 240: Left Kidney, 255:Spleen'
    )

    parser.add_argument(
        '--training_technologies',
        type=lambda s: tuple([str(item) for item in s.split(',')]),
        default=('CT', 'MR'),
        help='names of the training technologies files.'
    )
    parser.add_argument(
        '--training_modalities',
        type=lambda s: tuple([str(item) for item in s.split(',')]),
        default=(( ),('T1DUAL','T2SPIR')),
        help='names of the training modality files.'
    )

    parser.add_argument(
        '--local',
        type=bool,
        default=False,
        help='True if running in local and no GPU availables.' #Used for debug, it will cush when trying to load the model.
    )
    parser.add_argument(
        '--EWC',
        type=bool,
        default=False,
        help='Whether to use EWC to regularize parameters.'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=1,
        help='Number of epochs to use for training.'
    )

    parser.add_argument(
        '--nb_channels',
        type=int,
        default=1,
        help='Number of copies of the same data for the different channels. Need it for pretrained model with brats'
    )

    FLAGS, unparsed = parser.parse_known_args()

    if not FLAGS.data_file:
        print('Please specify the argument data_file')

    FLAGS.Base_directory = os.path.join(os.getcwd(),'Data_and_Pretrained_Models',FLAGS.Base_directory)
    FLAGS.data_file = os.path.join(FLAGS.Base_directory,FLAGS.data_file)

    #FLAGS.nb_channels = 1

    FLAGS.patch_shape = None
    FLAGS.truth_channel = FLAGS.nb_channels
    FLAGS.input_shape = tuple([FLAGS.nb_channels] + list(FLAGS.image_shape))
    FLAGS.augment = FLAGS.flip or FLAGS.distort
    if FLAGS.train_decoder_only:
        FLAGS.non_trainable_list = ['conv3d_' + str(s) for s in range(1, 15)]
    else:
        FLAGS.non_trainable_list = None

    f1 = lambda s: '_'.join([(item) for item in s.split(',')])
    f2 = lambda s: tuple([float(item) for item in s.split(',')])

    new_exp_name = os.path.join(FLAGS.Base_directory,'Trained_models')

    if FLAGS.model_file == 'None':
        new_exp_name = os.path.join(new_exp_name, 'Labels_' + f1(FLAGS.labels.replace(" ","")))

        if not FLAGS.transfer_model_file:
            new_exp_name = os.path.join(new_exp_name, 'No_Transfer')
        else:
            tr_model = os.path.relpath(FLAGS.transfer_model_file, os.getcwd())
            new_exp_name = os.path.join(new_exp_name,
                                        'Transfer')
        if FLAGS.ID != 'None':
            new_exp_name += FLAGS.ID
            print('Experiment with ' + FLAGS.ID + ' is run')

        if (not os.path.isdir(new_exp_name)) and (not FLAGS.return_name_only):
            os.makedirs(new_exp_name)
        elif FLAGS.return_name_only:
            pass
        else:
            print('The Experiment and the specified ID already exist.')
            raise EnvironmentError

        FLAGS.model_file = os.path.join(new_exp_name, 'Model.h5')
        FLAGS.logging_file = os.path.join(new_exp_name, 'Log.csv')

    if FLAGS.training_file == 'None':
        FLAGS.training_file = os.path.join(FLAGS.Base_directory, FLAGS.data_file+'_training_ids.pkl')
    if FLAGS.validation_file == 'None':
        FLAGS.validation_file = os.path.join(FLAGS.Base_directory, FLAGS.data_file+'_validation_ids.pkl')

    FLAGS.labels = f2(FLAGS.labels)
    FLAGS.n_labels = len(FLAGS.labels)

    print("Prediction for labels :", FLAGS.labels)
    if not FLAGS.return_name_only:
        main()
    else:
        print(FLAGS.model_file)