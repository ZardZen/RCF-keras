import numpy as np
from keras.optimizers import Adam, SGD
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import argparse
from loss import cross_entropy_balanced, pixel_error
from rcf import rcf
import os
from keras.utils import multi_gpu_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser(description='Keras Training')
    # ========= paths for training
    ap.add_argument("-npath", "--npy_path", default="data/", required=False,
                    help="path to npy. files to train")
    ap.add_argument("-mpath", "--model_path", default="model_save/", required=False,
                    help="path to save the output model")
    ap.add_argument("-lpath", "--log_path", default="log/", required=False,
                    help="path to save the 'log' files")
    ap.add_argument("-name","--model_name", default="rcf_our_Adam.h5", required=False,
                    help="output of model name")
    # ========= parameters for training
    ap.add_argument("-p", "--pretrain", default=0, required=False, type=int,
                    help="load pre-train model or not")
    ap.add_argument("-r", "--rows", default=320, required=False, type=int,
                    help="shape of rows of input image")
    ap.add_argument("-c", "--cols", default=480, required=False, type=int,
                    help="shape of cols of input image")
    ap.add_argument('-bs', '--batch_size', default=8, type=int,
                    help='batch size')
    ap.add_argument('-ep', '--epoch', default=500, type=int,
                    help='epoch')
    ap.add_argument('-m', '--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    args = vars(ap.parse_args())
    return args


def train(args):
    X_train = np.load(args["npy_path"] + 'imgs_BSDS.npy')
    #X_val = np.load(args["npy_path"] + 'X_val.npy')
    y_train = np.load(args["npy_path"] + 'imgs_mask_BSDS_B.npy')
    #y_val = np.load(args["npy_path"] + 'y_val.npy')
    if args["pretrain"]:
        model = load_model(args["model_path"] + args["model_name"],
                       custom_objects={'cross_entropy_balanced': cross_entropy_balanced, 'pixel_error': pixel_error})
    else:
        model = rcf(input_shape=(args["rows"], args["cols"], 3)))

    model.summary()
    lr_decay = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, verbose=1, min_lr=1e-6)
    checkpointer = ModelCheckpoint(args["model_path"] + args["model_name"], verbose=1, save_best_only=True)
    tensorboard = TensorBoard(log_dir=args["log_path"])
    callback_list = [lr_decay, checkpointer, tensorboard]

    #optimizer = SGD(lr=1e-5, momentum=args["momentum"], nesterov=False)
    optimizer = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999)
    #parallel_model = multi_gpu_model(model, gpus=1)
#     parallel_model.compile(loss={'o1': cross_entropy_balanced,
#                                 'o2': cross_entropy_balanced,
#                                 'o3': cross_entropy_balanced,
#                                 'o4': cross_entropy_balanced,
#                                 'o5': cross_entropy_balanced,
#                                 'ofuse': cross_entropy_balanced,
#                                 },
#                                 metrics={'ofuse': pixel_error},
#                                 optimizer=optimizer)
    model.compile(loss={'o1': cross_entropy_balanced,
                                'o2': cross_entropy_balanced,
                                'o3': cross_entropy_balanced,
                                'o4': cross_entropy_balanced,
                                'o5': cross_entropy_balanced,
                                'ofuse': cross_entropy_balanced,
                                },
                                metrics={'ofuse': pixel_error},
                                optimizer=optimizer)

#     RCF = model.fit(X_train, [y_train, y_train, y_train, y_train, y_train, y_train],
#                                 validation_data=(X_val, [y_val, y_val, y_val, y_val, y_val, y_val]),
#                                 batch_size=args["batch_size"], epochs=args["epoch"],
#                                 callbacks=callback_list, verbose=1)

    #RCF = parallel_model.fit(X_train, [y_train, y_train, y_train, y_train, y_train, y_train],validation_split=0.2,
     #                       batch_size=args["batch_size"], epochs=args["epoch"], callbacks=callback_list, verbose=1)
    RCF = model.fit(X_train, [y_train, y_train, y_train, y_train, y_train, y_train],validation_split=0.2,
                            batch_size=args["batch_size"], epochs=args["epoch"], callbacks=callback_list, verbose=1)


if __name__ == "__main__":
    args = args_parse()
    train(args)
