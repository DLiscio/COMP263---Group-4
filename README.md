# COMP263---Group-4: Evaluating Deep Neural Networks using the Histopathologic Cancer Detection dataset

## Introduction

The purpose of this project is to gain practical hands on experience by applying learned knowledge in Deep Learning concepts to a real-world project. We will utilize three different learning techniques: supervised, unsupervised, and transfer to conduct extensive experimentation on the dataset to develop capable models. We will also analyze each models performance in training and testing extensively, comparing results for one model to another in hopes to understand which model performs best as a solution within the datasets problem scope and potentially why. Results for training and testing have been stored in the "results" directory, and within a specified directory with the learning tecniques name, with data to be stored within the data directory (see **[Dataset](#dataset)**). 

## Dataset

The dataset used for this project was the Kaggle Histopathologic Cancer Detection dataset, which contains magnified partial images of larger pathology scans, containing both images of those with cancer and those without, used for a Kaggle machine learning contest. The dataset contains two directories with images, one for training and one for testing, a .csv file with labels for the training images, and a .csv file with a sample submission for the contest.

The dataset utilized for this project is too large to be hosted directly in this repository. You can download the dataset using the link below by signing into Kaggle and accepting the contest terms:

- **[Download the Histopathologic Cancer Detection dataset here](https://www.kaggle.com/c/histopathologic-cancer-detection/data)**

### **Setup Instructions**  

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

## Authors
- Damien Liscio
- Fan Yang
- Sophia Ojegba
- Harpreet Singh Dhanda
