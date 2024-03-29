from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

@app.route("/")
def index():
    # Load the dataset
    df = pd.read_csv('sentiment_analysis_results.csv')

    # Create a figure and axes for the subplot
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram of sentiment scores
    sentiment_scores = df['sentiment_score']
    axs[0].hist(sentiment_scores, bins=20, color='skyblue', edgecolor='black')
    axs[0].set_xlabel('Sentiment Score')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Histogram of Sentiment Scores')

    # Bar chart of sentiment categories
    sentiment_categories = []
    for score in df['sentiment_score']:
        if score > 0.05:
            sentiment_categories.append('positive')
        elif score < -0.05:
            sentiment_categories.append('negative')
        else:
            sentiment_categories.append('neutral')
    sentiment_categories = pd.Series(sentiment_categories)
    sentiment_counts = sentiment_categories.value_counts()
    axs[1].bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'red', 'blue'])
    axs[1].set_xlabel('Sentiment Category')
    axs[1].set_ylabel('Count')
    axs[1].set_title('Bar Chart of Sentiment Categories')

    # Save the plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    # Convert plot to base64 encoded string
    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Embed the plot in HTML
    return render_template('index.html', plot_data=plot_data)

if __name__=="__main__":
    app.run(debug=True)
