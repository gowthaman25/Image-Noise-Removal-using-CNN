# Image Noise Removal - CNN-keras     
A keras implemention of Deep CNN for Image Denoising

### Dependence
```
tensorflow
keras2
numpy
opencv
```
### Prepare train data

Clean patches are extracted from 'data/Train100' and saved in 'data/npy_data'.

### Train
```
$ python Main1.py -- trains the given image directory
```

Trained models are saved in current directory

### Test
```
$ python Main1.py -- test the given image directory('data/Test/Set68') by the trained model.
```

denoised images are saved in 'data/OutImg'.

### Results

The average PSNR(dB) and SSIM results are taken for the dataset.

Noise Level  --- 25 
PSNR     ------- 28.88  
SSIM     -------  0.80
