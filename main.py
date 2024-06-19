import joblib
import string
import nltk
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
best_rf_model = joblib.load('models/RandomForest_model.joblib')

import re
# Text preprocessing function
def preprocess_text(text):
    text = text.lower()  # Lowercase text
    text = re.sub(re.compile('<.*?>'),'',text) # Removing Html Tags
    text = re.sub('\[.*?\]', '', text) # Remove square brackets
    text = re.sub('https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # Remove punctuation
    text = re.sub('\n', '', text) # Remove new line (/n)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)# Remove non-alphanumeric characters
    # text = re.sub('\w*\d\w*', '', text)  # Removes Digit But in this case it make be Important
    tokens = word_tokenize(text)  # Tokenization
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return " ".join(tokens)

def predict_sentiment(text,tfidf_vectorizer,best_rf_model):
  text = preprocess_text(text)
  X_new = tfidf_vectorizer.transform([text]).toarray()
  prediction = best_rf_model.predict(X_new)
  if prediction[0] == 0:
    return 'Not a Disaster'
  elif prediction[0] == 1:
    return 'Disaster'

def main():
    st.title("Tweet Disaster Analysis")

    # Text input
    input_text = st.text_input("Enter the tweet for analysis:")

    # Button to submit the input
    if st.button("Predict Sentiment"):
        # Get the prediction
        output = predict_sentiment(input_text)
        # Display the output
        st.text(f"Predicted Sentiment: {output}")

if __name__ == "__main__":
    main()