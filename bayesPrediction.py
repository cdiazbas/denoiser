import numpy as np
import platform
import os
import time
import argparse
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py


# To deactivate warnings: https://github.com/tensorflow/tensorflow/issues/7778
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import astroNN
import tensorflow as tf
import network.bayes_model as nn_model
import keras.backend.tensorflow_backend as ktf
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
        self.model, self.model_prediction, mse_lin_ext, mse_var_ext = nn_model.model_regression_dropout_var()


        print("Loading weights... {0}_weights.hdf5".format(self.network_type))
        self.model.load_weights("{0}_weights.hdf5".format(self.network_type))

    
    def predict(self):
        print("Predicting validation data...")

        input_validation = np.zeros((1,self.nx,self.ny,1), dtype='float32')
        input_validation[0,:,:,0] = self.image

        # From our tests, the epistemic uncertainty is around one order of magnitude 
        # smaller. Therefore, one could make a single forward pass of the network to 
        # have a rough estimation of the total uncertainty (without the MonteCarlo). 

        start = time.time()
        result = np.array(self.model_prediction.predict(input_validation))
        sigma_total = np.sqrt(np.exp(result[0]))
        prediction = result[1]
        end = time.time()
        print("Prediction took {0:3.2} seconds...".format(end-start))        
        
        medio = 3*2.6e-3
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,6))
        plt.subplot(131)
        plt.title('Original')
        plt.imshow(imgs,cmap='seismic',origin='lower',interpolation='None',vmin=-medio,vmax=+medio)
        plt.minorticks_on(); plt.locator_params(axis='y', nbins=4); plt.ylabel('Y [pixel]'); plt.xlabel('X [pixel]')
        plt.subplot(132)
        plt.title('Output DNN')
        plt.imshow(prediction[0,:,:,0],cmap='seismic',vmin=-medio,vmax=+medio,origin='lower',interpolation='None')
        plt.minorticks_on(); plt.locator_params(axis='y', nbins=4); plt.xlabel('X [pixel]'); plt.tick_params(axis='y',labelleft=False)
        plt.subplot(133)
        plt.title('Uncertainty')
        plt.imshow(sigma_total[0,:,:,0]*1e3,cmap='gray_r',origin='lower',interpolation='None')        
        plt.minorticks_on(); plt.locator_params(axis='y', nbins=4); plt.xlabel('X [pixel]'); plt.tick_params(axis='y',labelleft=False)
        plt.savefig('docs/prediction'+str(self.number)+'.png',bbox_inches='tight')

        np.save(self.output,result)


     
if (__name__ == '__main__'):

    parser = argparse.ArgumentParser(description='Prediction')
    parser.add_argument('-i','--input', help='input',default='example_synthetic.npy')
    parser.add_argument('-o','--output', help='output',default='output/prediction_bayes_simulation.npy')
    parser.add_argument('-p','--picture', help='picture',default='_bayes')
    parser.add_argument('-m','--model', help='model', default='weights/simulation_bayes')
    parsed = vars(parser.parse_args())

    # Example to clean
    imgs = np.load(parsed['input'])

    out = deep_network(parsed['model'],parsed['output'],parsed['picture'])
    out.define_network(image=imgs)
    out.predict()
    # To avoid the TF_DeleteStatus message:
    # https://github.com/tensorflow/tensorflow/issues/3388
    ktf.clear_session()

    # python bayesPrediction.py -i example_sst.npy -o output/prediction_bayes_sst.npy -m weights/sst_bayes
    # python bayesPrediction.py -i example_synthetic.npy -o output/prediction_bayes_simulation.npy




