# -*- coding: utf-8 -*-

# Imports and versions

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
from scipy import ndimage
from scipy.io import loadmat
import nibabel as nib
import pickle as pk
from scipy import interpolate

# set random seed for testing (to have the same state in subsequent tests, for production run don't do it)
np.random.seed(30)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras import backend as K



"""# Read battery magnetometry data
skipping this for now, since not needed immediately, but just defining the coordinate grid compatible with the battery data that we have.
"""

# rscale and cscale as calculated from reading in real data and downsampling
# so that we do not need to read in experimental data right away
# check that can do without reading data
rscale=np.array([7.40000000e-05, 1.97920635e-03, 3.88441270e-03, 5.78961905e-03,
       7.69482540e-03, 9.60003175e-03, 1.15052381e-02, 1.34104444e-02,
       1.53156508e-02, 1.72208571e-02, 1.91260635e-02, 2.10312698e-02,
       2.29364762e-02, 2.48416825e-02, 2.67468889e-02, 2.86520952e-02,
       3.05573016e-02, 3.24625079e-02, 3.43677143e-02, 3.62729206e-02,
       3.81781270e-02, 4.00833333e-02, 4.19885397e-02, 4.38937460e-02,
       4.57989524e-02, 4.77041587e-02, 4.96093651e-02, 5.15145714e-02,
       5.34197778e-02, 5.53249841e-02, 5.72301905e-02, 5.91353968e-02,
       6.10406032e-02, 6.29458095e-02, 6.48510159e-02, 6.67562222e-02,
       6.86614286e-02, 7.05666349e-02, 7.24718413e-02, 7.43770476e-02,
       7.62822540e-02, 7.81874603e-02, 8.00926667e-02, 8.19978730e-02,
       8.39030794e-02, 8.58082857e-02, 8.77134921e-02, 8.96186984e-02,
       9.15239048e-02, 9.34291111e-02, 9.53343175e-02, 9.72395238e-02,
       9.91447302e-02, 1.01049937e-01, 1.02955143e-01, 1.04860349e-01,
       1.06765556e-01, 1.08670762e-01, 1.10575968e-01, 1.12481175e-01,
       1.14386381e-01, 1.16291587e-01, 1.18196794e-01, 1.20102000e-01])
cscale=np.array([0.001     , 0.00229032, 0.00358065, 0.00487097, 0.00616129,
       0.00745161, 0.00874194, 0.01003226, 0.01132258, 0.0126129 ,
       0.01390323, 0.01519355, 0.01648387, 0.01777419, 0.01906452,
       0.02035484, 0.02164516, 0.02293548, 0.02422581, 0.02551613,
       0.02680645, 0.02809677, 0.0293871 , 0.03067742, 0.03196774,
       0.03325806, 0.03454839, 0.03583871, 0.03712903, 0.03841935,
       0.03970968, 0.041     ])


"""# setting up the grid for the magnetic susceptibility"""

# battery dimensions
battery_dims=np.array([5,30,40])*1e-3;
dims=np.array([5,50,60])*1e-3;  # cell dimensions


### probe_dist=2.3e-2;   # 2 cm top and bottom
probe_dist=1.59e-2;   # 2 cm top and bottom

#probe_dist=1e-2;   # 2 cm top and bottom

npts=[1,16,32];

# volume element
# will this be accurate or is it minus 1?
dV=np.prod(dims/npts);   # this is volume per point in the susceptibility map, seems the correct way

# some recentering of coordinates based on experimental data
# for first data
centery=0.021;
centerz=0.06;

# for second data
centery=0.015;
centerz=0.077;

# for damaged cell data
centery=0.020;
centerz=0.065;

# for new send data
centery=0.021;
centerz=0.062;

#field_dims=[dims(2)*3, dims(3)*3];
field_dims=np.array([60,80])*1e-3;


# see here: this volume probably calculated incorrectly, but it's also probably not needed
field_npts=[20,30];
dVfield=np.prod(field_dims/field_npts);

"""### convert both the magnetic susceptibility grid positions and the magnetic field positions into lists `[x1,y1,z1; x2,y2,z2; etc ]`
`src_pos_list` is the list for magnetic susceptibility, and `field_pos_list` is the one for magnetic field.
"""

# this way bottom of cell starts at x=0 (+padding), so the probe_dist is measured from the bottom of cell
srcpos=[[],[],[]]
for i in range(3):
    srcpos[i]=np.linspace(0,dims[i],npts[i]+2)
    srcpos[i]=srcpos[i][1:(npts[i]+1)]
srcpos[1]=srcpos[1]+centery-dims[1]/2
srcpos[2]=srcpos[2]+centerz-dims[2]/2

srcxv,srcyv,srczv=np.meshgrid(srcpos[0],srcpos[1],srcpos[2],indexing='ij')

src_fulllength=np.prod(npts)
src_pos_list=np.concatenate((srcxv.reshape((src_fulllength,1)),srcyv.reshape((src_fulllength,1)),srczv.reshape((src_fulllength,1))),axis=1)

rv, cv = np.meshgrid(rscale, cscale, indexing='ij')  # ij indexing produces same shape as newy, newz

# create field-pos / amp vectors
# remember that for two field components, I would have to stack them, so maybe best for now to keep them separate (position vs. field meas)
fulllength=np.prod(rv.shape)
# y, z, newy, newz

field_pos_list=np.concatenate((cv.reshape((fulllength,1)),rv.reshape((fulllength,1))),axis=1)
field_pos_list=np.insert(field_pos_list,0,probe_dist,axis=1)


"""## set up conversion matrix between magn. susceptibility and magn. field
so that we have $\mathbf{B}=A \cdot \mathbf{m}$ as a matrix-vector multiplication
"""

# modified for multidim to include multiple field components (y-z)
# now modif for x-y-z source dims

oneD=0   # to do z-only calc in this framework for magnetic susceptibility (may be more stable)

# make sure to reshape such that multiply the correct field components
# A matrix is not very sparse, so maybe faster to do in non-sparse setup
fpl=field_pos_list.shape[0]
A=np.zeros((2,fpl,src_pos_list.shape[0],3),dtype=float)
for i in range(src_pos_list.shape[0]):
    posdiff=src_pos_list[i,:]-field_pos_list
      # this is a broadcasted operation
      # posdiff is field_pos_list but every row is subtracted from ths ith row in src_pos_list
    inv_r=1/np.sqrt(np.sum(posdiff**2,axis=1))
      # performed for each row
    inv_r5=inv_r**5
    inv_r3=inv_r**3
    
    for fidx in range(2):
        fidx2=fidx+1    # this is the real dim index (compatible with sidx)
                        # since I only have y and z components of the field
        
        if oneD:
            sidx=2
            A[fidx,:,i,sidx]=3*posdiff[:,fidx2]*posdiff[:,sidx]*inv_r5
            if sidx==fidx2:
                A[fidx,:,i,sidx]=A[fidx,:,i,sidx]-inv_r3
        else:
            for sidx in range(3):
                A[fidx,:,i,sidx]=3*posdiff[:,fidx2]*posdiff[:,sidx]*inv_r5
                if sidx==fidx2:
                    A[fidx,:,i,sidx]=A[fidx,:,i,sidx]-inv_r3
    
A=A.reshape((fpl*2,src_pos_list.shape[0]*3))


# proper conversion units

B0=20e-6
A=A*dV*B0/4/np.pi


"""## left out SVD and pseudoinverse calculation here since not immediately relevant

## Model setup
model based on papers (Bollman, etc) with some modifications, links given
"""

# for regularization
dropout_level=0.15

#adapted from https://colab.research.google.com/drive/1ltjXmi6fSAe4YBgJrmHH_wjTl9VxFRgl

def get_figure():
    """
    Returns figure and axis objects to plot on. 
    Removes top and right border and ticks, because those are ugly
    """
    fig, ax = plt.subplots(1)
    plt.tick_params(top=False, right=False, which='both') 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig, ax

def conv_block_h2(input_tensor, num_filters):
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Dropout(dropout_level)(encoder)
    encoder = layers.Activation('relu')(encoder)
    return encoder

def encoder_block_h2(input_tensor, num_filters):
    encoder = conv_block_h2(input_tensor, num_filters)
    encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
    return encoder_pool, encoder

def decoder_block_h2(input_tensor, concat_tensor, num_filters):
    decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Dropout(dropout_level)(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    return decoder

# define the loss function for TF, for now just differentce beetween predicted and true magnetic fields
def custom_loss(y_true,y_pred):
    print("y_pred shape:",y_pred[0,:,:,0].shape)
    #penalty=batt_mask128(y_pred[0,:,:,0])
    #loss=K.mean(K.square(y_pred-y_true),axis=-1)+np.sum(0*np.abs(penalty)) #can adjust the penalty weight
    loss=K.mean(K.square(y_pred-y_true),axis=None)  #+K.sum(0*K.abs(penalty)) #can adjust the penalty weight
    return loss

inputs_h2 = layers.Input(shape=(64,32,2))
encoder0_pool_h2, encoder0_h2 = encoder_block_h2(inputs_h2, 8)
encoder1_pool_h2, encoder1_h2 = encoder_block_h2(encoder0_pool_h2, 16)
encoder2_pool_h2, encoder2_h2 = encoder_block_h2(encoder1_pool_h2, 32)
encoder3_pool_h2, encoder3_h2 = encoder_block_h2(encoder2_pool_h2, 64)
center_h2 = conv_block_h2(encoder3_pool_h2, 128)
decoder3_h2 = decoder_block_h2(center_h2, encoder3_h2, 64)
decoder2_h2 = decoder_block_h2(decoder3_h2, encoder2_h2, 32)
decoder1_h2 = decoder_block_h2(decoder2_h2, encoder1_h2, 16)
# figure out how to downsample filter sizes, simple with conv, but do I need any activations? 
# removing one block gives output of fact 2 smaller each side
# but what happens to concatenation, probably should still use it, maybe in the filter reduction?
# actually this concatenation may not be needed
# does concatenation need the same number of filters?
# maybe could even do without concatenation?
# decoder0_h2 = decoder_block_h2(decoder1_h2, encoder0_h2, 8)
outputs_h2 = layers.Conv2D(3, (1, 1), padding="same")(decoder1_h2)   # simply set number of output channels here, seems legit

model_ht2b = models.Model(inputs=[inputs_h2], outputs=[outputs_h2])

adam=keras.optimizers.Adam(beta_2=0.99)

model_ht2b.compile(optimizer=adam,
             loss=custom_loss)

model_ht2b.summary()

"""## Generating Training Set
generating fake magnetic susceptibility distributions and calculating the expected magnetic fields from this based on known A matrix
"""

num_sim2=200 #can adjust, higher means slower but more accurate

#training_labels=np.zeros((num_sim2, npts[1], npts[2],3))
############
# for network best to create the susceptibility in the transposed version
training_labels=np.zeros((num_sim2, npts[2], npts[1],3))   
training_data=np.zeros((num_sim2, rv.shape[0], rv.shape[1],2))

def calcfield(suscept):
    source_vec=np.squeeze(suscept[:,:,:]).reshape((src_fulllength*3,1))
    magfield=np.dot(A,source_vec)
    fieldy = magfield[0:fpl,0].reshape((rv.shape[0],rv.shape[1]))
    fieldz = magfield[fpl:2*fpl,0].reshape((rv.shape[0],rv.shape[1]))
    return fieldy, fieldz

# generate random magntic susceptibility distributions, based on a set of random gaussian peaks 
maxlevelrange=200e-6
numberpeaks=10;
idx1=range(npts[2])
idx2=range(npts[1])
midx1,midx2=np.meshgrid(idx1,idx2,indexing='ij')
for ii in range(num_sim2):
    # for now just produce z susceptibility (easier for checking result?)
    ############
    
    # here provide alternative training set  exp(-x^2/(2sigma^2))
    if True:
        for iii in range(numberpeaks):
            pos1=np.random.rand(1)*npts[2]
            pos2=np.random.rand(1)*npts[1]
            w1=np.random.rand(1)*npts[2]/5+1
            w2=np.random.rand(1)*npts[1]/5+1
            amp=np.random.rand(1)*maxlevelrange
            training_labels[ii, :, :,2]=training_labels[ii, :, :,2]+amp*np.exp(-((midx1-pos1)/w1)**2-((midx2-pos2)/w2)**2)
    else:
        # for network best to create the susceptibility in the transposed version
        #training_labels[ii, :, :,2] = maxlevelrange*np.random.rand(npts[1], npts[2])
        training_labels[ii, :, :,2] = maxlevelrange*np.random.rand(npts[2], npts[1])  # only z susceptibility for now 
        
    training_data[ii, :, :,0],training_data[ii, :, :,1] = calcfield(training_labels[ii,:,:,:])

# TF requires this kind of transformation into tensor
train_images_t2b=tf.constant(training_data)
train_labels_t2b=tf.constant(training_labels)


def imshow_center(data):
    maxval=np.max(np.abs(data))
    plt.imshow(data, cmap="seismic",vmin=-maxval,vmax=maxval)
    plt.colorbar()

imshow_center(train_images_t2b[33,:,:,0])
plt.savefig('training_sample.png')

# for historical reasons, the magnetic susceptibility map shows up transpose, 
# I guess it would be good to change that at some point, but for now keeping it 
imshow_center(np.transpose(train_labels_t2b[10,:,:,2]))
plt.savefig('label_sample.png')

"""# Training the model"""

# I read that using adam with learning rate 0.001 is good
num_epochs=10 #can adjust ultimately should be quite large
history_ht2b = model_ht2b.fit(train_images_t2b, train_labels_t2b,  epochs=num_epochs, batch_size=5, shuffle=True)

# plot loss fn vs. epochs

#adapted from https://colab.research.google.com/drive/1ltjXmi6fSAe4YBgJrmHH_wjTl9VxFRgl

loss_history_ht2b = history_ht2b.history['loss']

fig, ax = get_figure()

startpoints=0
ax.plot((np.arange(num_epochs*1)+1)[startpoints:], loss_history_ht2b[startpoints:], marker="o", linewidth=2, color="orange", label="loss1")
ax.set_xlabel('epoch')
ax.legend(frameon=False);

# ___ epochs until convergence

fig.savefig('history.png')

# save in one step
model_ht2b.save('model.h5')

# save history (if needed separately)
with open('train_history.db', 'wb') as file_pi:
        pk.dump(loss_history_ht2b, file_pi)

"""# Loading Saved Model and predicting"""

# just for testing that I can read in model and proceed from here
#del model_ht2b
#del history_ht2b
del loss_history_ht2b

loss_history_ht2b = pk.load(open('train_history.db', "rb"))
fig, ax = get_figure()
num_epochs=len(loss_history_ht2b)
startpoints=1900
ax.plot((np.arange(num_epochs*1)+1)[startpoints:], loss_history_ht2b[startpoints:], marker="o", linewidth=2, color="orange", label="loss1")
ax.set_xlabel('epoch')
ax.legend(frameon=False);

# load in one step
# fixed with adding custom_loss function, in future, better to save model and weights separately?
model_ht2b=tf.keras.models.load_model('model.h5',custom_objects={'custom_loss': custom_loss})

# do I need to recompile after loading?
model_ht2b.compile(optimizer=adam,
             loss=custom_loss)

#Predicting the training data, adapted from https://colab.research.google.com/drive/1ltjXmi6fSAe4YBgJrmHH_wjTl9VxFRgl

test_patch_nbr = 20
susax=2
X_test = train_images_t2b[np.newaxis,test_patch_nbr,:,:,:]     # why do I need to add new axis for prediction set?
y_pred_ht2 = model_ht2b.predict(X_test)
plt.figure()
imshow_center(y_pred_ht2[0,:,:,susax]-train_labels_t2b[test_patch_nbr,:,:,susax])
yf,zf=calcfield(y_pred_ht2[0,:,:,:])
# yf and zf end up too large
plt.figure()
imshow_center(np.squeeze(X_test[0,:,:,0])-yf)
plt.figure()
imshow_center(np.squeeze(X_test[0,:,:,1])-zf)
#plt.savefig('prediction.png')
