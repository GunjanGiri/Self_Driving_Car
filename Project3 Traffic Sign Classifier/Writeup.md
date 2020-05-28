

**Build a Traffic Sign Recognition Project**

The goals/steps of this project are the following:
* Load the dataset (train.p , test.p , valid.p)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

All the  Images are Presented in the traffic-sign-images folder.

## Rubric Points

 
### Dataset Exploration

#### Dataset Summary

I used the vanilla python commands and numpy library to calculate summary statistics of the traffic signs data set:

* Number of images in training set is 34799
* Number of images in validation set is 4410
* Number of images in test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### Exploratory Visualization

Below is an exploratory visualization of the data set. The bar charts depict distribution of images per class in training, testing and validation datasets. It is clear that there is a wide variation in number of images available per class in each of these datasets. Therefore, it is logical to think that data augmentation using preprocessing techniques can help reduce this disparity and thereby result in improved performance of the network.

Images are in the Folder.


### Design and Test a Model Architecture

#### Preprocessing

The original training dataset was first converted to grayscale because traffic signs can be uniquely distinguished by shapes in most cases. Moreover, the reference paper uses grayscale images to achieve a respectable classification accuracy. Normalization was later applied to the dataset to achieve zero mean and equal variance. This helps the optimization algorithm converge in lesser steps thereby saving computational time.

As can be seen from the visualization of test, training and validation data; the number of images per class varies greatly in each of the datasets. This means that there might be features in validation data, that are not fully encompassed in the training data. To overcome this issue, data augmentation is applied. The first step in augmentation is to make a dataset of images that are below a certain number in each of the classes. Specifically, classes with number of images less than 800 are selected to be augmented. For this purpose, images are randomly selected from each class and various transformations like random zoom, random translation, random rotation and histogram equalization are applied. Also, data is generated is done a way such that the final distribution of class vs count is relatively flat. Finally, the generated data is converted to grayscale and normalized before concatenating with the original training dataset.


The probabilities to get images from each category during training became equal. It dramatically improved neural network performance. The size of training set became 43 * 2010 = 86430 samples.

#### Model Architecture

The model architecture is defined in *Step 2: Design and Test a Model Architecture*, *Model Architecture* subsection.
The architecture has 5 layers - 2 convolutional and 3 fully connected.
It is LeNet-5 architecture with only one modification - dropouts were added between the layer #2 and layer #3, the last convolutional layer and the first fully connected layer. It was done to prevent neural network from overfitting and significantly improved its performance as a result.

Below is the description of model architecture.

| Layer                 |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 32x32x1 gray scale image                      | 
| Convolution 5x5       | 1x1 stride, same padding, outputs 28x28x6     |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, outputs 14x14x6                   |
| Convolution 5x5       | 1x1 stride, same padding, outputs 10x10x16    |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, outputs 5x5x16                    |
| Flatten               | output 400                                    |
| Drop out              |                                               |
| Fully connected       | output 120                                    |
| RELU                  |                                               |
| Fully connected       | output 84                                     |
| RELU                  |                                               |
| Fully connected       | output 43                                     |





The decision to select LeNet as a base structure was made because performance of LeNet for classifying characters is well documented. As traffic signs are also made of simple shapes like characters, one can reasonably assume that LeNet would give satisfactory performance. 


#### Model Training

The model is using Adam optimizer to minimize loss function. It worked better than stochastic gradient descent. 
The following hyperparameters were defined and **carefully** adjusted:
```
# learning rate; with 0.001, 0.0009 and 0.0007 the performance is worse 
RATE       = 0.0008

# number of training epochs; here the model stops improving; we do not want it to overfit
EPOCHS     = 30

# size of the batch of images per one train operation; surprisingly with larger batch sizes neural network reached lower performance
BATCH_SIZE = 128

# the probability to drop out the specific weight during training (between layer #2 and layer #3)
KEEP_PROB  = 0.7

# standart deviation for tf.truncated_normal for weights initialization
STDDEV     = 0.01
```

#### Solution Approach

My final model results were:
* training set accuracy of 99%
* validation set accuracy of 96% 
* test set accuracy of 93.5%

To keep things manageable at the beginning, LeNet architecture (implemented in the LeNet lab) was selected. However, the validation accuracy was limited to 89-94%. Moreover, plotting the loss and accuracy plots revealed overfitting. To overcome this issue, two dropout layers were added. The keep probability for these dropout layers was tuned by trial and error. The loss and accuracy plots for the current architecture are given below. It can be observed that the transition of accuracy and loss as the number of epochs increase is smooth.

As mentioned earlier, the performance of LeNet for classifying characters is well documented. Also traffic signs are made of simple shapes like characters from which one can reasonably assume that LeNet would give satisfactory performance. Moreover, the final model results verify that the model is indeed working well.

### Test a Model on New Images

#### Acquiring New Images

Images are Present in the traffic-sign-images Folder. You can go through them.

The first image, "right of way at the next intersection", might be difficult to classify because it has a picture in the middle that is not very visible at the resolution 32x32. 
The last image may be difficult to classify because it contains text in the middle with a specific speed limit. 
There are a lot of categories in the "speed limit" super-category. 
At the 32x32 resolution and low brightness or contrast, they are hardly distinguishable.

#### Performance on New Images

Surprisingly, on these five images, the performance of the predictions was 100%. However, when the similar model was trained with 50 epochs, there was a mistake with "speed limit 30 km/h" traffic sign, the model was overfitted.
There may be mistakes on other types of images. With other models, I had a problem with "end of all speed and passing limits" traffic sign classification. Also, the results on the test set were not perfect (93.5%), so, certainly, there are images somewhere on the web that this model will not be able to recognize.

The code can be found in the section *Step 3: Test a Model on New Images*, *Load and Output the Images* subsection.

Here are the results of the prediction:

|                  PREDICTED                  |                   ACTUAL                    |
|:-------------------------------------------:|:-------------------------------------------:|
| 1            Speed limit (30km/h)           | 1            Speed limit (30km/h)           |
| 12              Priority road               | 12              Priority road               |
| 11  Right-of-way at the next intersection   | 11  Right-of-way at the next intersection   |
| 14                   Stop                   | 14                   Stop                   |
| 15               No vehicles                | 15               No vehicles                |

#### Model Certainty - Softmax Probabilities

The code for making predictions on my final model is located in the *Step 3: Test a Model on New Images*, *Top 5 Softmax Probabilities For Each Image Found on the Web* subsection.

The model was quite certain about the four images. It was also pretty certain about "no vehicles" traffic signal, but not totally. Below are top 5 softmax probabilities for "**no vehicles**" traffic sign.

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| 0.756710649           | No vehicles                                   | 
| 0.122047313           | Speed limit (30km/h)                          |
| 0.0512931943          | Priority road                                 |
| 0.0313977301          | Stop                                          |
| 0.0146821784          | No passing                                    |
