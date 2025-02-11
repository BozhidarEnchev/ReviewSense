# Sentiment Analysis on IMDB Movie Reviews

This repository contains a sentiment analysis model that classifies IMDB movie reviews as either positive or negative. The model is implemented using Keras and TensorFlow, leveraging LSTM (Long Short-Term Memory) networks for text classification.

## Overview

The sentiment analysis model works by processing IMDB movie reviews and predicting whether the sentiment behind the review is positive or negative. The model is built using the following steps:

- Preprocessing text (removal of stopwords, punctuation, and tokenization)
- LSTM-based architecture for classification
- Model training on a labeled dataset of IMDB reviews

### Key Features:
- Uses IMDB movie review dataset
- Text preprocessing (cleaning and tokenization)
- LSTM for sentiment classification
- Keras/TensorFlow implementation

## Dataset

You need the **IMDB Dataset** to train or test the model. You can download it from Kaggle:

1. Visit the [IMDB Dataset of 50k movie reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) page.
2. Download the dataset as a `.zip` file and extract the contents.
3. Place the `IMDB Dataset.csv` file in the `data/` directory within this repository.

> **Note**: The `IMDB Dataset.csv` file contains the reviews along with sentiment labels (`positive`/`negative`).

## Setup

### 1. Clone the Repository

First, clone the repository to your local machine:

```
git clone https://github.com/BozhidarEnchev/ReviewSense.git
cd ReviewSense 
```

### 2. Install the requirements
```
pip install -r requirements.txt
```
### 3. Download the Dataset
As mentioned above, download the dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and extract the `IMDB Dataset.csv` file to the `data/` directory.

### 4. Train the model
This will:
- Load the data
- Preprocess the text (tokenization and padding)
- Train the model with LSTM layers
- Save the trained model as sentiment_model.keras

```
python sentiment_analysis.py
```
>**Note**: It is not necessary to train the model because the repository comes with a pre-trained model (`sentiment_model.keras`). You can directly use the pre-trained model for predictions.
### 5. Use the model for prediction
To classify a review using the pre-trained model, run the following command:
```
python predict.py
```
Once the model is loaded, you will be prompted to input a review comment. The model will classify it as either Positive or Negative based on the sentiment of the comment.
