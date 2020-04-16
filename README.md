# Unet for Ultrasonic Image Segmentation
　Unet used for medical image is very robust and effective. In this tiny project, I use Unet to identify the arm's median nerve on the ultrasonic imgae.  Most of the code is based on the implementation of [Unet](https://github.com/zhixuhao/unet) by zhixuhao
 on Python 3, Keras, and TensorFlow. In the following results, color red was labeled by experts and color green was the results inference by our trained model.The color yellow indcate the overlap ofred and green.
(note: Actually, the results via the inference of unet model are binary image; the following results is via post processing and then show it.)


![Alt text](https://github.com/YunaLiou/UltraSonic-Image-Unet/blob/master/readme/Demo.png)




　If you want to use the simple code,please download this repository and further read the following tutorial.



## Putting your datasets
　There are 3 folders (**image**, **ground_truth**, **test_image**) under the folder,datasets. The folder,**image**, was used for placing the images for training.  The folder,**ground_truth**, was used for putting the labeled image corresponding to the images for training. ( Note: you need to be sure the files' order in both of folders (**image**, **ground_truth**) is consistent. Or, you can name the image and its corresponding labeled image with the same name and further put them into the 2 folders,respectively.)

The folder,**test_image**, was used for placing those image you want to predict.


The data directory strutures are as following:

    datasets  
    　　└ image
       　　 └ 0000.png
       　　 └ 0001.png
       　　 └ ...
    　　└ ground_truth
        　　└ 0000.png
      　　  └ 0001.png
       　　 └ ...
    　　└ test_image
        　　└ 
        　　└ 
       　　 └ 



## Training your datasets
To start training on your own dataset, the things you need to do modify in this repo is:

* main in [unet.py](https://github.com/YunaLiou/UltraSonic-Image-Unet/blob/master/unet.py):

If you just want to run on CPU rather than GPU,you need to comment out the two lines.

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

* main in [unet.py](https://github.com/YunaLiou/UltraSonic-Image-Unet/blob/master/unet.py):

You need to modify the 256 and 512 in the following code to your width and height of images.
   (note: Be sure that your all images have the single size.)

    myunet = Unet(img_rows=256, img_cols=512)


When completing the traing process, the model wiil be saved as a hdf5 file and it will use for predicting your images in the folders,test_image.

The result inferenced by trained model will be save as npy files and futher save as image into the folder, resultimage.

## Semantic Segmentation Samples on Ultrasonic Image
　The following demonstration is sequence of ultrasonic images which have label (green) and prediction (red).
![image](https://github.com/YunaLiou/UltraSonic-Image-Unet/blob/master/readme/Demo2.gif)

