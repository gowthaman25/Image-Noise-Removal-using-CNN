import numpy as np 
import glob
import cv2
import pandas as pd
import tensorflow as tf
import tensorflow.keras 
from keras.models import load_model,Sequential
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import Adam
from skimage.measure import compare_psnr, compare_ssim
from keras.layers import Input,BatchNormalization,Subtract,Conv2D,Lambda,Activation
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from cnn_model import cnn_model

src_dir = '/content/drive/My Drive/code/data/Train400/'
file_list = glob.glob(src_dir+'*.png')  # get name list of all .png files

batch_size=128
pat_size=40
stride=10
step=0
scales=[1,0.9,0.8,0.7]
count=0
#calculate the number of patches
for i in range(len(file_list)):
    img = cv2.imread(file_list[i],0) 
    for s in range(len(scales)):
        newsize=(int(img.shape[0]*scales[s]),int(img.shape[1]*scales[s]))
        img_s=cv2.resize(img,newsize,interpolation=cv2.INTER_CUBIC)
        im_h,im_w=img_s.shape
        for x in range(0+step,(im_h-pat_size),stride):
            for y in range(0+step,(im_w-pat_size),stride):
                count +=1

origin_patch_num=count
if origin_patch_num % batch_size !=0:
    numPatches=(origin_patch_num/batch_size +1)*batch_size
else:
    numPatches=origin_patch_num
print('filelist=%d ,total patches=%d, batch_size=%d, total_batches=%d' % (len(file_list),numPatches, batch_size, numPatches/batch_size))

#numpy array to contain patches for training
inputs=np.zeros((int(numPatches), int(pat_size), int(pat_size),1),dtype=np.uint8)

#generate patches
count=0
for i in range(len(file_list)):
    img = cv2.imread(file_list[i],0) 
    for s in range(len(scales)):
        newsize=(int(img.shape[0]*scales[s]),int(img.shape[1]*scales[s]))
        img_s=cv2.resize(img,newsize,interpolation=cv2.INTER_CUBIC)
        img_s=np.reshape(np.array(img_s,dtype="uint8"),
                (img_s.shape[0],img_s.shape[1],1))
        im_h,im_w,_ = img_s.shape
        for x in range(0+step,im_h-pat_size,stride):
            for y in range(0+step,im_w-pat_size,stride):
                inputs[count,:,:]=img_s[x:x+pat_size,
                    y:y+pat_size]
                count += 1

#pad the batch
if count < numPatches:
    to_pad=int(numPatches-count)
    inputs[-to_pad:,:,:,:]=inputs[:to_pad,:,:,:]
np.save("img_clean_pats.npy",inputs)

# load the data and normalize it
cleanImages=np.load('img_clean_pats.npy')
print(cleanImages.dtype)
cleanImages=cleanImages/255.0
cleanImages=cleanImages.astype('float32')
data = cleanImages

epoch = 5
save_every = 2
lr = 1e-3
sigma = 25

psnr_val = []
ssim_val = []
name =[]
test_fol = '/content/drive/My Drive/code/data/Test/Set68/'
out_dir = '/content/drive/My Drive/code/data/Outimg/'

def data_aug(img, mode=0):
    
    if mode == 0:
        return img
        
def gen_patches(file_name):

    # read image
    img = cv2.imread(file_name, 0)  # gray scale
    h, w = img.shape
    scales = [1, 0.9, 0.8, 0.7]
    patches = []

    for s in scales:
        h_scaled, w_scaled = int(h*s),int(w*s)
        img_scaled = cv2.resize(img, (h_scaled,w_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches
        for i in range(0, h_scaled-patch_size+1, stride):
            for j in range(0, w_scaled-patch_size+1, stride):
                x = img_scaled[i:i+patch_size, j:j+patch_size]
                # data aug
                for k in range(0, aug_times):
                    x_aug = data_aug(x, mode=np.random.randint(0,8))
                    patches.append(x_aug)
    
    return patches

def train_datagen(y_,batch_size=8):
    indices = list(range(y_.shape[0]))
    while(True):
        np.random.shuffle(indices)    # shuffle
        for i in range(0, len(indices), batch_size):
            ge_batch_y = y_[indices[i:i+batch_size]]
            noise =  np.random.normal(0, sigma/255.0, ge_batch_y.shape)    # noise
            ge_batch_x = ge_batch_y + noise  # input image = clean image + noise
            yield ge_batch_x, ge_batch_y

def lr_scheduler(epoch, lr):
    decay_rate = 0.1
    decay_step = 90
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

def train():
    models.compile(optimizer=Adam(), loss=['mse'])
    callbacks=[LearningRateScheduler(lr_scheduler)]
    # use call back functions
    #ckpt = ModelCheckpoint('/model_{epoch:02d}.h5', monitor='val_loss',verbose=0, period=save_every)
    #csv_logger = CSVLogger('/log.csv', append=True, separator=',')
    history = models.fit_generator(train_datagen(data, batch_size=batch_size),
                    steps_per_epoch=len(data)//batch_size, epochs=epoch, verbose=0, 
                    callbacks=callbacks)
    models.save('myModel.h5')
    return models 

def test(models):
    out_dir = '/content/drive/My Drive/code/data/Outimg/'
    psnr_val = []
    ssim_val = []
    test_dir = glob.glob(test_fol+'*.png')
    for t in range(len(test_dir)):
        print('test dir',len(test_dir))
        img_clean = cv2.imread(str(test_dir[t]),0)
        img_test = np.array(img_clean,dtype='float32')/255
        noise =  np.random.normal(0, sigma/255.0, img_test.shape)    # noise
        img_test = img_test.astype('float32')
        # predict
        x_test = img_test.reshape(1, img_test.shape[0], img_test.shape[1], 1) 
        y_predict = models.predict(x_test)
        # calculate numeric metrics
        img_out = y_predict.reshape(img_clean.shape)
        img_out = np.clip(img_out, 0, 1)
        img_out = np.array((img_out*255).astype('uint8')) 
        filename = (str(test_dir[t])).split('/')[-1].split('.')[0]    # get the name of image file
        cv2.imwrite(out_dir+str(t)+'.png',img_out)
        psnr_noise, psnr_denoised = compare_psnr(img_clean, img_test), compare_psnr(img_clean, img_out)
        ssim_noise, ssim_denoised = compare_ssim(img_clean, img_test), compare_ssim(img_clean, img_out)
        psnr_val.append(psnr_denoised)
        ssim_val.append(ssim_denoised)
    if len(psnr_val) != 0 :    
        psnr_avg = sum(psnr_val)/len(psnr_val)
    else :
        psnr_avg = 0
    if len(ssim_val) != 0 :
        ssim_avg = sum(ssim_val)/len(ssim_val)
    else :
        ssim_avg = 0
    psnr_val.append(psnr_avg)
    ssim_val.append(ssim_avg)
    print('Average PSNR = {0:.2f}, SSIM = {1:.2f}'.format(psnr_avg, ssim_avg))
    return psnr_val , ssim_val

if __name__=='__main__':
    models = cnn_model()
    models = train()
    psnr_val,ssim_val = test(models)

    
