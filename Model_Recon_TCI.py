

## Import the necessary Libraries
import numpy as np
import sklearn.metrics as metrics
from keras.layers import Input, Add, Activation, Dropout, Flatten, Dense,Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
import keras.initializers
from keras.optimizers import Adam
from Model_recon import *
import keras.callbacks as callbacks
import keras.utils.np_utils as kutils
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
import scipy.io as sio 
import scipy.linalg as linalg
from keras import backend as K
import os
from keras.layers import Lambda
batch_size =512
nb_epoch = 100
img_rows, img_cols = 32, 32
from keras.callbacks import ModelCheckpoint


M=256
PHI=np.random.randn(M,32*32)
PHI=linalg.orth(PHI.T).T
PHI_ex=PHI.T
n=2
X_in=Input(shape=(32,32,1,))
layers,a=bulid_reconstruction(X_in,PHI_ex,M,n)

model = Model(inputs=[X_in], outputs=[layers[0],;ayers[1],layers[2])

print(model.summary())


def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def psnr(noise,X_in):
    mse = tf.losses.mean_squared_error(labels=X_in,predictions=noise)
    def loss(y_true, y_pred):
        
        psnr_e=20 * log10(255.0 / tf.math.sqrt(mse))
        return psnr_e

    return loss


model.compile(loss=['mse','mse','mse'],loss_weights=[0.33,0.33,0.33],optimizer=Adam(lr=1e-3),metrics=[psnr(A,X_in)])

## Data Loading
Training_Image=sio.loadmat('C:\\RAFI_SHARED\\CS_NET\\GAN\\Training_Image_Label.mat')['X_out']
Testing_Image=sio.loadmat('C:\\RAFI_SHARED\\CS_NET\\GAN\\validation_Image_Label.mat')['X_out']
Training_Image=np.reshape(Training_Image,(Training_Image.shape[0],32,32,1))
Testing_Image=np.reshape(Testing_Image,(Testing_Image.shape[0],32,32,1))
X_train=Training_Image
X_test=Testing_Image
idx=np.random.randint(50000,size=50000)
P=idx[:40000]
Q=idx[40000:]
TI=Training_Image[P,:]
VI=Training_Image[Q,:]
X_train=TI
X_val=VI


model.fit([X_train], [X_train,X_train,X_train],epochs=100,batch_size=32,
          validation_data=([X_val],[X_val,X_val,X_val] ))


## Inference Phase


epoch=100
model.save_weights(
            'params_256_learned_TCI_new_epoch_{0:03d}.hdf5'.format(epoch), True)
layer_name = 'activation_6' #Layer correspoing to the final output


# In[14]:


import time

intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
tic = time.clock()
intermediate_output = intermediate_layer_model.predict(X_test)
toc = time.clock()
print(toc-tic)
imgs, row, col,_= intermediate_output.shape
def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse)),math.sqrt(mse)

Y_dev=X_test
Y_recon=intermediate_output
J=np.zeros((imgs,1))
T=np.zeros((imgs,1))
for i in range(imgs):
        J[i],T[i]=(psnr((Y_recon[i]),(Y_dev[i])))
print(np.mean(J))
print(np.mean(T))





