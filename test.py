import numpy as np
from keras.models import load_model
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from loss import cross_entropy_balanced, pixel_error
import argparse
import os
import cv2 as cv

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
path="results"

def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser(description='Keras Training')
    # ========= paths for training
    ap.add_argument("-npath", "--npy_path", default="data/",
                    help="path to npy. files to train")
    ap.add_argument("-mpath", "--model_path", default="model_save/",
                    help="path to save the output model")
    ap.add_argument("-name","--model_name", default="rcf_our_Adam.h5",
                    help="output of model name")
    ap.add_argument("-r", "--rows", type=int, default=320,
                    help="shape of rows of input image")
    ap.add_argument("-c", "--cols", type=int, default=480,
                    help="shape of cols of input image")
    args = vars(ap.parse_args())
    return args


def test(args):
    X_train = np.load(args["npy_path"] + 'data_test.npy')
    #X_train = np.load(args["npy_path"] + 'imgs_BSDS.npy')
    #X_test = np.load(args["npy_path"] + 'X_test.npy')
    #X_val = np.load(args["npy_path"] + 'X_val_ori.npy')
    y_train = np.load(args["npy_path"] + 'label_test.npy')
    #y_train = np.load(args["npy_path"] + 'imgs_mask_BSDS_B.npy')
    #y_test = np.load(args["npy_path"] + 'y_test.npy')
    #y_val = np.load(args["npy_path"] + 'y_val_concat.npy')

    model = load_model(args["model_path"] + args["model_name"],
                       custom_objects={'cross_entropy_balanced': cross_entropy_balanced, 'pixel_error': pixel_error})
    
    # test all images from test.npy
    print(len(X_train))
    for i in range(5):5
        Y_pred = model.predict(X_train[i].reshape((-1, 320, 480, 3)))
        print(len(Y_pred))
        for j in range(len(Y_pred)):
            y_pred=Y_pred[j]
            y_pred = y_pred.reshape((320, 480))
            out = cv.normalize(y_pred, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
            cv.imwrite(path+'/'+'pred_'+format(str(i))+'_'+format(str(j)) + '.jpg',out)
            plt.figure(figsize=(25, 16))
            plt.subplot(1, 3, 1)
            plt.imshow(X_train[i], cmap='binary')
            plt.subplot(1, 3, 2)
            print(y_train[i].shape)
            plt.imshow(y_train[i].reshape((320, 480)), cmap='binary')
            plt.subplot(1, 3, 3)
            plt.imshow(y_pred, cmap='binary')
            name = path+'/'+str(i) + '_'+str(j)+'.jpg'
            plt.savefig(name)
            plt.close()

if __name__ == "__main__":
    from Subpixel import SubpixelConv2D
  
    args = args_parse()
    test(args)