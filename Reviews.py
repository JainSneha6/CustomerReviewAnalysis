import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import seaborn as sns

class DataPreprocessor:
    def __init__(self, filename):
        self.filename = filename
    
    def load_data(self):
        return pd.read_csv(self.filename)
    
    def clean_text(self, text):
        if pd.isnull(text):
            return ''
        text = text.lower()
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = word_tokenize(text)
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        return ' '.join(words)
    
    def preprocess_data(self, df):
        df.drop(columns=self.columns_to_drop, inplace=True)
        df.dropna(subset=self.text_columns, inplace=True)
        for column in self.text_columns:
            df[column] = df[column].apply(self.clean_text)
        return df
    
    def initialize_nltk(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def setup(self):
        self.initialize_nltk()
        self.columns_to_drop = ['Clothing ID', 'Age', 'Recommended IND', 'Positive Feedback Count', 'Division Name',
                                'Department Name', 'Class Name']
        self.text_columns = ['Title', 'Review Text']

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def get_sentiment_scores(self, text):
        return self.analyzer.polarity_scores(text)['compound']
    
    def classify_sentiment(self, score):
        if score >= 0.05:
            return 'Positive'
        elif score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

class SentimentClassifier:
    def __init__(self):
        pass
    
    def convert_rating_to_sentiment(self, rating):
        if rating == 5 or rating == 4:
            return 'Positive'
        elif rating == 1 or rating == 2:
            return 'Negative'
        else:
            return 'Neutral'
    
    def calculate_accuracy(self, actual_sentiment, predicted_sentiment):
        return accuracy_score(actual_sentiment, predicted_sentiment)

class SVMClassifier:
    def __init__(self):
        self.classifier = SVC(kernel='linear', random_state=42)
        self.vectorizer = TfidfVectorizer(max_features=5000)
    
    def train(self, X_train, y_train):
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        self.classifier.fit(X_train_tfidf, y_train)
    
    def predict(self, X_test):
        X_test_tfidf = self.vectorizer.transform(X_test)
        return self.classifier.predict(X_test_tfidf)

class NBClassifier:
    def __init__(self):
        self.classifier = MultinomialNB()
        self.vectorizer = TfidfVectorizer(max_features=5000)
    
    def train(self, X_train, y_train):
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        self.classifier.fit(X_train_tfidf, y_train)
    
    def predict(self, X_test):
        X_test_tfidf = self.vectorizer.transform(X_test)
        return self.classifier.predict(X_test_tfidf)

class Visualizer:
    def __init__(self):
        pass
    
    def plot_histogram(self, data):
        plt.hist(data, bins=20, color='skyblue', edgecolor='black')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Frequency')
        plt.title('Histogram of Sentiment Scores')
        plt.show()
    
    def plot_bar_chart(self, x, y):
        plt.figure(figsize=(8, 6))
        sns.barplot(x=x, y=y, palette='viridis')
        plt.xlabel('Sentiment Category')
        plt.ylabel('Count')
        plt.title('Bar Chart of Sentiment Categories')
        plt.show()

# Usage example
if __name__ == "__main__":
    preprocessor = DataPreprocessor('AD.csv')
    preprocessor.setup()
    df = preprocessor.load_data()
    df = preprocessor.preprocess_data(df)
    
    sentiment_analyzer = SentimentAnalyzer()
    df['sentiment_score'] = df['Review Text'].apply(sentiment_analyzer.get_sentiment_scores)
    df['sentiment'] = df['sentiment_score'].apply(sentiment_analyzer.classify_sentiment)
    
    sentiment_classifier = SentimentClassifier()
    df['actual_sentiment'] = df['Rating'].apply(sentiment_classifier.convert_rating_to_sentiment)
    accuracy = sentiment_classifier.calculate_accuracy(df['actual_sentiment'], df['sentiment'])
    print("Accuracy:", accuracy)
    
    svm_classifier = SVMClassifier()
    X_train, X_test, y_train, y_test = train_test_split(df['Review Text'], df['sentiment'], test_size=0.2, random_state=42)
    svm_classifier.train(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("SVM Classifier Accuracy:", accuracy)
    
    nb_classifier = NBClassifier()
    nb_classifier.train(X_train, y_train)
    y_pred = nb_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Naive Bayes Classifier Accuracy:", accuracy)
    
    visualizer = Visualizer()
    visualizer.plot_histogram(df['sentiment_score'])
    visualizer.plot_bar_chart(df['sentiment'].value_counts().index, df['sentiment'].value_counts().values)
