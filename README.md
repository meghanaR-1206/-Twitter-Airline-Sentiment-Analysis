# -Twitter-Airline-Sentiment-Analysis
This project leverages machine learning techniques to predict the sentiment of tweets related to US airlines. With over 14,000 labeled tweets from the Twitter US Airline Sentiment Dataset, the task is to classify these tweets as either positive, negative, or neutral based on the content of the tweet. The project demonstrates end-to-end sentiment analysis workflow, from data preprocessing to model evaluation.

Key Steps:
Data Preprocessing:

Cleaned the tweet text by removing URLs, hashtags, mentions, and punctuation.
Tokenized the text and performed lemmatization.
Removed stopwords to retain important words for sentiment classification.
Feature Extraction:

Utilized TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to convert text into numerical features that can be used by machine learning models.
Model Training:

Logistic Regression: A popular algorithm for binary and multiclass classification tasks, used here to train a model on the processed data.
Support Vector Machine (SVM): A powerful classifier used to maximize the margin between classes, effective in text classification problems.
Model Evaluation:

Evaluated the models using various metrics such as accuracy, precision, recall, and F1-score to ensure good model performance.
Splitted the data into training and validation sets to prevent overfitting.
Prediction:

Predicted the sentiment of tweets in the test dataset and saved the results in a CSV file format as required.
Tools and Libraries:
Scikit-learn for model building and evaluation.
NLTK for text preprocessing, tokenization, and stopword removal.
Pandas for data manipulation and handling CSV files.
