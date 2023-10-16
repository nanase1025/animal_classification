# animal_classification
❗️❗️❗️ English | [中文文档](https://github.com/nanase1025/animal_classification/blob/main/README_zh.md)

This is a baseline of the animal classification.
This project is mainly about classifying animal pictures. The file is divided into training set and test set. The image labels of the training set come from the folder name to which they belong. For example, the labels of all images in the ape folder are ape, and the test set is strictly prohibited from participating in training.

## main processes
### Main processes and documents that need to be retained:
1. Write the data processing code dataset.py so that pictures and labels can correspond one-to-one to facilitate subsequent training.
2. Write the training code train.py, use the training set train to save the model parameters model.path after multiple rounds of training, and select the number of training rounds yourself.
3. Write the test file inference.py, load the model parameter model.path, and output the label of each picture in the test set test, and organize it in the form of output.json file. A sample Json file is as follows:
{
    "0": "cat",
    "1": "ape",
    "2": "fish"
}

## About the model.py
I used 3 kinds of model(resnet, vgg, googlenet)
