# Image-Classifier
In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice, you'd train this classifier, then export it for use in your application. We'll be using this dataset of 102 flower categories.

When you've completed this project, you'll have an application that can be trained on any set of labelled images. Here your network will be learning about flowers and end up as a command line application. 

## Prerequisites
1. The Code is written in **Python 3** . If you don't have Python installed you can find it [here](https://www.python.org/downloads/).
Ensure you have the latest version of pip.
2. Additional Packages that are required are: [Numpy](http://www.numpy.org/), [Pandas](https://pandas.pydata.org/), [MatplotLib](https://matplotlib.org/), [Pytorch](https://pytorch.org/), PIL and json. You can donwload them using [pip](https://pypi.org/project/pip/):
    - ```pip install numpy pandas matplotlib pil```<br/>
    or [conda](http://www.numpy.org/)
    - ```conda install numpy pandas matplotlib pil```
    
**NOTE**: In order to install Pytorch follow the instructions given on the official site.

## Command Line Application

- Train a new network on a data set with **train.py**
  - Basic Usage : ```python train.py data_directory```<br/>
  - Prints out current epoch, training loss, validation loss, and validation accuracy as the netowrk trains
  - Options:
    - Set direcotry to save checkpoints: ```python train.py data_dor --save_dir save_directory```
    - Choose arcitecture (densenet121 or vgg16 available): ```python train.py data_dir --arch "vgg16"```
    - Set hyperparameters: ```python train.py data_dir --learning_rate 0.001 --hidden_layer1 120 --epochs 20```
    - Use GPU for training: ```python train.py data_dir --gpu gpu```
  - Output: A trained network ready with checkpoint saved for doing parsing of flower images and identifying the species.
    
- Predict flower name from an image with **predict.py** along with the probability of that name. That is you'll pass in a single image /path/to/image and return the flower name and class probability
  - Basic usage: ```python predict.py /path/to/image checkpoint```
  - Options:
    - Return top K most likely classes: ```python predict.py input checkpoint ---top_k 3```
    - Use a mapping of categories to real names: ```python predict.py input checkpoint --category_names cat_To_name.json```
    - Use GPU for inference: ```python predict.py input checkpoint --gpu```

## Data
The data used specifically for this assignment are a flower database(.json file). It is not provided in the repository as it's larger than what github allows.<br/>
The data need to comprised of 3 folders:
1. test
2. train 
3. validate<br/>

Generally the proportions should be 70% training 10% validate and 20% test.

Inside the train, test and validate folders there should be folders bearing a specific number which corresponds to a specific category, clarified in the json file. For example if we have the image x.jpg and it is a lotus it could be in a path like this /test/5/x.jpg and json file would be like this {...5:"lotus",...}. 

## GPU/CPU
As this project uses deep CNNs, for training of network you need to use a GPU. However after training you can always use normal CPU for the prediction phase.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
Credits must be given to Udacity for providing starting code for this project. The data was provided by Udacity partners at Bertelsmann Arvato Analytics.
