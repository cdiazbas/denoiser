import numpy as np
import platform
import os
import time
import argparse
# from astropy.io import fits
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py


# To deactivate warnings: https://github.com/tensorflow/tensorflow/issues/7778
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
import network.models as nn_model
import matplotlib as mpl
mpl.use('Agg')


# To reduce number of cores used in the inference
reducecore = False
if reducecore:
    ncores = 2
    from keras import backend as K
    K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=ncores, inter_op_parallelism_threads=ncores)))



class deep_network(object):

    def __init__(self, network_type, output, number):

# Only allocate needed memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        session = tf.Session(config=config)
        ktf.set_session(session)

        self.network_type = network_type
        self.number = number
        self.output = output
        self.nfilter = 32

    def define_network(self, image):
        print("Setting up network...")

        self.image = image
        self.nx = image.shape[0]
        self.ny = image.shape[1]
        self.model = nn_model.unet(start_ch=self.nfilter)

        print("Loading weights... {0}_weights.hdf5".format(self.network_type))
        self.model.load_weights("{0}_weights.hdf5".format(self.network_type))

    
    def predict(self):
        print("Predicting validation data...")

        input_validation = np.zeros((1,self.nx,self.ny,1), dtype='float32')
        input_validation[0,:,:,0] = self.image


        start = time.time()
        out = self.model.predict(input_validation)
        end = time.time()
        print("Prediction took {0:3.2} seconds...".format(end-start))        
        
        ima = self.image
        medio = 3*2.6e-3
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,6))
        plt.subplot(131)
        plt.title('Stokes Q - Original')
        plt.imshow(ima,cmap='seismic',origin='lower',interpolation='None',vmin=-medio,vmax=+medio)
        plt.minorticks_on(); plt.locator_params(axis='y', nbins=4)
        plt.subplot(132)
        plt.title('Stokes Q - Output DNN')
        plt.imshow(out[0,:,:,0],cmap='seismic',vmin=-medio,vmax=+medio,origin='lower',interpolation='None')
        plt.minorticks_on(); plt.locator_params(axis='y', nbins=4)
        plt.subplot(133)
        plt.title('Difference')
        plt.imshow(ima-out[0,:,:,0],cmap='seismic',vmin=-medio,vmax=+medio,origin='lower',interpolation='None')        
        plt.minorticks_on(); plt.locator_params(axis='y', nbins=4)
        plt.savefig('docs/prediction'+str(self.number)+'.png',bbox_inches='tight')

        np.save('output/prediction_sst.npy',out[0,:,:,0])


     
if (__name__ == '__main__'):

    parser = argparse.ArgumentParser(description='Prediction')
    parser.add_argument('-o','--output', help='output')
    parser.add_argument('-n','--number', help='number',default='_sst')
    parser.add_argument('-m','--model', help='model', default='weights/network_sst')
    parsed = vars(parser.parse_args())

    # f = fits.open(parsed['input'])
    # dire = parsed['dir']
    # print('Dir:'+dire)
    # input_file = dire+'database_test.h5'
    # f = h5py.File(input_file, 'r')
    # ntx, nx, ny, nlambda = f['stokes'].shape
    # nindex = 1
    # imgs = f['stokes'][nindex:nindex+1,:,:,:].astype('float32')
    # ntx, nx, ny, nq = f['cube'].shape        
    # # salida = f['cube'][0:1,:,:,:].astype('float32')
    # f.close()
    # from train import generaflat
    
    imgs = np.load('example_sst.npy')

    out = deep_network(parsed['model'],parsed['output'],parsed['number'])
    out.define_network(image=imgs)
    out.predict()
    # To avoid the TF_DeleteStatus message:
    # https://github.com/tensorflow/tensorflow/issues/3388
    ktf.clear_session()

    # os.system('python3 plot3Ddef.py -dir '+dire)
    # os.system('python3 plotError.py')





