# Goodreads Reviews Sentiment Analyser

## Purpose:
Derive an aggregate sentiment score of a book to gain a more accurate and objective measure of reader satisfaction, attained through textual analysis.

## How it works:
- Loads reviews from a CSV file.
- Inserts them into an SQLite database.
- Analyzes the reviews using VaderSentiment and normalizes the score between 1 to 10.
    (1 being most negative sentiment and 10 being most positive sentiment.)
- Updates the sentiment scores in the database and prints it.

## Run it

```bash
pip install -r requirements.txt
python GoodreadsReviewSentimentAnalyser.py
```

The script runs on 50 sample reviews (10 each for five books) in `data/sample_reviews.csv` and prints every review with its sentiment score.

## Result:
In manual testing of a select number of books, I identified misrepresented star ratings for 20% of the analyzed books.

## Future Work:
- Increase the number of reviews used to analyze each book.
- Validate the VaderSentiment model for accuracy and take it's shortcomings into account for margin of error.
- Create a user friendly application for better accessibility.
