import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'text'])
df = df.dropna()

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2)

# transform text to numbers (term-frequency-inverse document frequency)
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)

# training with Multinomial Naive Bayes, using probability it guesses if it's spam or not based on the transformation above
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

X_test_tfidf = vectorizer.transform(X_test)
predictions = model.predict(X_test_tfidf)

print(f"\nAccuracy: {accuracy_score(y_test, predictions)}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))


results_df = pd.DataFrame({
    'message': X_test,
    'realLabel': y_test,
    'predictedLabel': predictions
})


false_positives = results_df[(results_df['realLabel'] == 'ham') & (results_df['predictedLabel'] == 'spam')]

false_negatives = results_df[(results_df['realLabel'] == 'spam') & (results_df['predictedLabel'] == 'ham')]

print("\nFalse Positives (Real messages sent to Spam):")
i = 0
if false_positives.empty:
    print("No False Positives")
else:
    for index, row in false_positives.iterrows():
        print(f"{i+1}: {row['message']}")
        i += 1

print("\nFalse Negatives (Spam messages that passed):")
j = 0
if false_negatives.empty:
    print("No False Negatives")
else:
    for index, row in false_negatives.iterrows():
        print(f"{j+1}: {row['message']}")
        j += 1