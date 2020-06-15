# Image Noise Removal - CNN-keras     
A keras implemention of Deep CNN for Image Denoising and analysed the effective of changing padding and strides in results of PSNR and SSIM. 

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
$ python Main.py -- trains the given image directory
```

Trained models are saved in current directory

### Test
```
$ python Main.py -- test the given image directory('data/Test/Set68') by the trained model.
```

denoised images are saved in 'data/OutImg'.

### Results

For the Noise Level 25 , average PSNR(dB) and SSIM results are taken for the dataset.

| Epoch   | Noise Level| Padding | Stride  | PSNR    | SSIM    |
|:-------:|:----------:|:-------:|:-------:|:-------:|:-------:|
| 10      | 25         |50       |5        |25.57    | 0.79    |
| 10      | 25         |60       |5        |26.16    | 0.75    |
| 10      | 25         |70       |4        |27.99    | 0.77    |
| 10      | 25         |40       |4        |28.62    | 0.77    |
| 10      | 25         |60       |3        |26.24    | 0.76    |
| 10      | 25         |10       |3        |28.38    | 0.75    |
| 10      | 25         |10       |1        |28.65    | 0.74    |
| 10      | 25         |**30**   |**1**    |**28.88**| **0.80**|

  ###  The above table shows the average PSNR and SSIM obtained for various padding size and strides. Here i have used minimum epochs with lesser gaussian noise. From the result, it shows that with the minimum Stride and increased Padding will helps to achieve maximum accuracy in image denoising. 
 

