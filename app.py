from flask import Flask, render_template, request
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

# Initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to classify sentiment of a custom string
def classify_sentiment(custom_string):
    # Get sentiment score for the custom string
    sentiment_score = analyzer.polarity_scores(custom_string)['compound']

    # Classify sentiment based on the sentiment score
    if sentiment_score >= 0.05:
        return 'Positive'
    elif sentiment_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        custom_string = request.form['custom_string']
        predicted_sentiment = classify_sentiment(custom_string)
        return render_template('result.html', custom_string=custom_string, sentiment=predicted_sentiment)

if __name__ == '__main__':
    app.run(debug=True)
