# COMP263---Group-4: Evaluating Deep Neural Networks using the Histopathologic Cancer Detection dataset

## Introduction

The purpose of this project is to gain practical hands on experience by applying learned knowledge in Deep Learning concepts to a real-world project. We will utilize three different learning techniques: supervised, unsupervised, and transfer to conduct extensive experimentation on the dataset to develop capable models. We will also analyze each models performance in training and testing extensively, comparing results for one model to another in hopes to understand which model performs best as a solution within the datasets problem scope and potentially why. Results for training and testing have been stored in the "results" directory, and within a specified directory with the learning tecniques name, with data to be stored within the data directory (see **[Dataset](#dataset)**). 

## Dataset

The dataset used for this project was the Kaggle Histopathologic Cancer Detection dataset, which contains magnified partial images of larger pathology scans, containing both images of those with cancer and those without, used for a Kaggle machine learning contest. The dataset contains two directories with images, one for training and one for testing, a .csv file with labels for the training images, and a .csv file with a sample submission for the contest.

The dataset utilized for this project is too large to be hosted directly in this repository. You can download the dataset using the link below by signing into Kaggle and accepting the contest terms:

- **[Download the Histopathologic Cancer Detection dataset here](https://www.kaggle.com/c/histopathologic-cancer-detection/data)**

### **Model Training Instructions**  

1. Download the dataset from Kaggle  
2. Extract the contents to the `data/` directory. discarding of the `sample_submission.csv` and keeping `train/`, `test/`, and `test_labels.csv`.
3. To run the files locally, execute the following commands from the `COMP263---Group-4-main/` directory:
  - **Supervised**:
  ```python
  python supervised.py
  ```
  - **Unupervised**:
  ```python
  python unsupervised.py
  ```
  - **State of the Art**:
  ```python
  python sota_model.py
  ```

## Web Application

The web application features visual implementations of all 3 learning types and offers the ability to see the trained models abilities. The supervised & state-of-the-art pages allow you to randomly select an image from the dataset, and uses the model to predict the images class and displays the prediction along with the actual class label. The unsupervised page allows the user to generate an image using the model, and then displays the generated image next to an actual image from the dataset for comparison. The web application is built using Flask and features an app.py containing page & model functionality endpoints.

To start the server, execute the following command from the `COMP263---Group-4-main/` directory:
  ```python
  python frontend/app.py
  ```

## Authors
- Damien Liscio
- Fan Yang
- Sophia Ojegba
- Harpreet Singh Dhanda
