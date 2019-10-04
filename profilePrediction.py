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


def crisp_load(fname, verb = True):
    import lptools as lp
    f1 = fname.format('')
    f2 = fname.format('_sp')
    nx, ny, ndum, nstokes, dtype, dum1 = lp.lphead(f1, verb)
    nw, nt, ndum, nstokes, dtype, dum1 = lp.lphead(f2, verb)
    io = np.memmap(f1, shape=(nt,nstokes,nw,ny,nx), offset=512,
                   dtype= dtype, mode='r')
    return io


# ncores = 10
# from keras import backend as K
# K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=ncores, inter_op_parallelism_threads=ncores)))


class deep_network(object):

    def __init__(self, inputFile, depth, model, activation, output, number):

        # Only allocate needed memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        session = tf.Session(config=config)
        ktf.set_session(session)

        self.input = inputFile
        self.depth = depth
        self.network_type = model
        self.activation = activation
        self.number = number
        self.output = output
        self.nfilter = 32

        self.model = nn_model.unet(start_ch=self.nfilter)
        print("Loading weights... {0}_weights.hdf5".format(self.network_type))
        self.model.load_weights("{0}_weights.hdf5".format(self.network_type))

    def define_network(self, image):
        self.image = image
        self.nx = image.shape[1]
        self.ny = image.shape[2]
            
    def predict(self,numerotime):
        print("Inferring clean data...")

        input_validation = np.zeros((self.image.shape[0],self.nx,self.ny,1), dtype='float32')
        input_validation[:,:,:,0] = self.image

        start = time.time()
        out = self.model.predict(input_validation)
        end = time.time()
        print("Prediction took {0:3.2} seconds...".format(end-start))        
        

        ima = self.image
        # medio = (np.abs(ima.min()) + np.abs(ima.max()))/5.
        medio = 3*2.6e-3
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,6))
        plt.subplot(131)
        plt.imshow(ima[8,:,:],cmap='seismic',origin='lower',interpolation='None',vmin=-medio,vmax=+medio)
        plt.subplot(132)
        plt.imshow(out[8,:,:,0],cmap='seismic',vmin=-medio,vmax=+medio,origin='lower',interpolation='None')
        plt.subplot(133)
        plt.imshow(ima[8,:,:]-out[8,:,:,0],cmap='seismic',vmin=-medio,vmax=+medio,origin='lower',interpolation='None')        
        plt.savefig('ssprediction'+str(numerotime)+'.pdf',bbox_inches='tight')
        
        return ima[:,:,:]-out[:,:,:,0]

   

if (__name__ == '__main__'):

    parser = argparse.ArgumentParser(description='Prediction')
    parser.add_argument('-i','--input', help='input', default=0)
    parser.add_argument('-o','--out', help='out')
    parser.add_argument('-n','--number', help='number')
    parser.add_argument('-d','--depth', help='depth', default=2)
    parser.add_argument('-m','--model', help='model',  default='weights/network_sst')
    parser.add_argument('-c','--activation', help='Activation', choices=['relu', 'elu'], default='relu')
    # parser.add_argument('-a','--action', help='action', choices=['cube', 'movie'], default='cube')
    parser.add_argument('-dir','--dir', help='dir', default='/scratch/')
    parsed = vars(parser.parse_args())


    dir0 = '/scratch/.../'
    fname = dir0 + 'crispex.stokes....time_corrected{0}.fits'

    import mfits
    datos = mfits.readfits(fname)
    nt,ns,nw,nx,ny = datos.shape
    datos = datos[:,:,:,:988,:944]
    print(datos.shape)
    
    sc = 2.6e-3

    import CRISpy.SaveLoad as crispy

    nmapa = np.zeros((datos.shape[0],datos.shape[1],datos.shape[2],datos.shape[3],datos.shape[4]))


    out = deep_network('{0}'.format(parsed['input']), depth=int(parsed['depth']), model=parsed['model'], 
                    activation=parsed['activation'], output=parsed['out'], number=parsed['number'])

    # Execute the network N times more:    
    nciclos = 0
    for istokes in [1,2,3]:
        # for jj in [1]:
        for jj in range(datos.shape[0]):
            print('==> time frame:', jj)
            
            input0 = datos[jj,istokes,:,:,:]*sc
            numerotime = str(jj)+'_'+str(istokes) 
            out.define_network(image=input0)
            ciclo = out.predict(numerotime)

            for i in range(nciclos):
                out.define_network(image=ciclo)
                ciclo = out.predict(numerotime)

            medio = 3*2.6e-3
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12,6))
            plt.subplot(131)
            plt.title('Original')
            plt.imshow(input0[8,:,:],cmap='seismic',origin='lower',interpolation='None',vmin=-medio,vmax=+medio)
            plt.minorticks_on(); plt.locator_params(axis='y', nbins=4); plt.ylabel('Y [pixel]'); plt.xlabel('X [pixel]')
            plt.subplot(132)
            plt.title('Output DNN')
            plt.imshow(input0[8,:,:]-ciclo[8,:,:],cmap='seismic',vmin=-medio,vmax=+medio,origin='lower',interpolation='None')
            plt.minorticks_on(); plt.locator_params(axis='y', nbins=4); plt.xlabel('X [pixel]'); plt.tick_params(axis='y',labelleft=False)
            plt.subplot(133)
            plt.title('Difference')
            plt.imshow(ciclo[8,:,:],cmap='seismic',vmin=-medio,vmax=+medio,origin='lower',interpolation='None')        
            plt.minorticks_on(); plt.locator_params(axis='y', nbins=4); plt.xlabel('X [pixel]'); plt.tick_params(axis='y',labelleft=False)
            # plt.savefig('ssprediction'+str(numerotime)+'.pdf',bbox_inches='tight')
            plt.savefig('docs/prediction'+str(numerotime)+'.png',bbox_inches='tight')
            output0 = (input0[:,:,:]-ciclo[:,:,:])/sc

            nmapa[jj,istokes,:,:,:] = output0


    # import sys
    # sys.exit()

    nmapa[:,0,:,:,:] = datos[:,0,:,:,:]
    print('All done')

    # To avoid the TF_DeleteStatus message:
    # https://github.com/tensorflow/tensorflow/issues/3388
    ktf.clear_session()

    # crispy.save_lpcube(nmapa, 'test',sp=True)
    mfits.writefits('test.fits', nmapa)



