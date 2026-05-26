Spam Detector (Machine Learning Project)

A Python-based machine learning project that classifies text messages as either "spam" or "ham" (regular messages) using Natural Language Processing and the Naive Bayes algorithm.


This project takes a raw dataset of text messages, cleans the data, translates the text into mathematical weights using TF-IDF (Term Frequency-Inverse Document Frequency), and trains a machine learning model to accurately identify spam.


At the end of the training, it also displays what False Negatives and False Positives, in order to analyze what the model got wrong.


Features:

- Data Preprocessing: Handles empty lines and formats the dataset.
    
- Text Vectorization: Converts text into numerical values using TF-IDF.
    
- Classification Model: Uses the Naive Bayes algorithm, which is a standard for text classification.
    
- Performance Metrics and Error Analysis: Outputs the accuracy and confusion matrix of the result as well as what entries the model got wrong.
    

Requirements:

This project requires Python 3.x and the following libraries:
- pandas
- scikit-learn

Make sure to also download the SMSSpamCollection file too, and put in the same folder as main.py, that is the dataset that the projects work on, by default.


How to use:

After downloading main.py and SMSSpamCollection.txt files and putting them into the same folder, and downloading the libraries, open main.py into your IDE of choice and run the python script, it should display in the console, in order: The accuracy, the confusion matrix, False Positives and False Negatives.

Github link: https://github.com/PadeanuTudor/Spam_Detector
