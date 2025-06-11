import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pickle
import re

# Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('UpdatedResumeDataSet.csv')

# Clean the resume text
print("Cleaning resume text...")
df['Resume'] = df['Resume'].apply(cleanResume)

# Prepare the data
X = df['Resume']
y = df['Category']

# Create and fit the label encoder
print("Encoding categories...")
le = LabelEncoder()
y = le.fit_transform(y)

# Save the label encoder
with open('encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# Create and fit the TF-IDF vectorizer
print("Creating TF-IDF vectors...")
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(X)

# Save the TF-IDF vectorizer
with open('tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
print("Training the model...")
svc_model = SVC(kernel='linear')
svc_model.fit(X_train, y_train)

# Save the model
print("Saving the model...")
with open('clf.pkl', 'wb') as f:
    pickle.dump(svc_model, f)

# Evaluate the model
print("Evaluating the model...")
train_score = svc_model.score(X_train, y_train)
test_score = svc_model.score(X_test, y_test)

print(f"Training accuracy: {train_score:.2f}")
print(f"Testing accuracy: {test_score:.2f}")
print("Model training completed and saved as 'clf.pkl'") 