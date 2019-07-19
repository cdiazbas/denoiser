from keras.layers import Dense
from keras.models import Model, Sequential, Input
from keras.layers import Dense, InputLayer, Dropout, Activation, regularizers
from keras import initializers
from keras.optimizers import Adam
from keras.backend import learning_phase, function
import keras.backend as K

# To get plot_model works, you need to install graphviz and pydot_ng
from keras.utils import plot_model
from astroNN.nn.layers import MCDropout, MCSpatialDropout2D
import astroNN
print(astroNN.__version__)
import tensorflow as tf

from astroNN.nn.losses import mse_lin_wrapper, mse_var_wrapper, robust_mse
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from keras.layers import Lambda, Multiply, Add


# ==================================================================================
def model_regression_dropout_var(filtros, dropout_rate, l2_rate, activation='relu', n_filters=64, l2_reg=1e-12,
        input_channel_num=1, out_ch=1, start_ch=32, inc_rate=2., depth=2, 
        batchnorm=False, maxpool=True, upconv=True, residual=False,disableDo=False):
    # UNet: code from https://github.com/pietz/unet-keras

    def _conv_block(m, dim, acti, bn, res, do=0):
        n = Conv2D(dim, 3, padding='same',kernel_initializer='he_normal', activation=acti)(m)
        n = MCDropout(do)(n)
        n = Conv2D(dim, 3, padding='same',kernel_initializer='he_normal', activation=acti)(n)
        n = MCDropout(do)(n)
        return Concatenate()([m, n]) if res else n

    def _level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
        if depth > 0:
            n = _conv_block(m, dim, acti, bn, res,do)
            m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
            m = _level_block(m, int(inc * dim), depth - 1, inc, acti, do, bn, mp, up, res)
            if up:
                m = UpSampling2D()(m)
                m = Conv2D(dim, 2, activation=acti, padding='same')(m)
            else:
                m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
            n = Concatenate()([n, m])
            m = _conv_block(n, dim, acti, bn, res,do)
        else:
            m = _conv_block(m, dim, acti, bn, res, do)
        return m

 
    def _level_block2(m, dim, depth, inc, acti, do, bn, mp, up, res):
        m = _conv_block(m, dim, acti, bn, res, do)
        m = _conv_block(m, dim, acti, bn, res, do)
        m = _conv_block(m, dim, acti, bn, res, do)
        m = _conv_block(m, dim, acti, bn, res, do)
        return m
    
    input_tensor = Input(shape=(None, None, 1),name='input')
    labels_err_tensor = Input(shape=(None,None, 1), name='label_err')

    # Good old output
    o1 = _level_block(input_tensor, 32, depth, inc_rate, activation, dropout_rate, batchnorm, maxpool, upconv, residual)
    linear_output = Conv2D(1,1,activation="linear",name='linear_output')(o1)
    
    # Data-dependent uncertainty outainty
    o2 = _level_block2(input_tensor, 32, 0, inc_rate, activation, dropout_rate, batchnorm, maxpool, upconv, residual)
    o2 = Conv2D(1,1, activation="linear")(o2)
    # We had to force to start close to the solution because working in log has this problem
    variance_output = Lambda(lambda x: -K.abs(x)-12,name='variance_output')(o2)
        
    model = Model(inputs=[input_tensor, labels_err_tensor], outputs=[variance_output, linear_output])
    model_prediction = Model(inputs=input_tensor, outputs=[variance_output, linear_output])
    
    mse_var_ext = mse_var_wrapper(linear_output, labels_err_tensor)
    mse_lin_ext = mse_lin_wrapper(variance_output, labels_err_tensor)

    return model, model_prediction, mse_lin_ext, mse_var_ext

