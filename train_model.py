import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from preprocess import clean_text

df = pd.read_csv("data/resumes.csv")

df['cleaned'] = df['Resume'].apply(clean_text)

X = df['cleaned']
y = df['Category']

vectorizer = TfidfVectorizer(max_features=3000)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

pickle.dump(model, open("model/resume_classifier.pkl", "wb"))
pickle.dump(vectorizer, open("model/tfidf.pkl", "wb"))

print("Model trained successfully")

