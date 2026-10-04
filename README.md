# Goodreads Reviews Sentiment Analyser

## Purpose:
Derive an aggregate sentiment score of a book to gain a more accurate and objective measure of reader satisfaction, attained through textual analysis.

## How it works:
- Scrapes reviews from the Goodreads website.
- Inserts initial scraped data into an SQLite database.
- Analyzes the reviews using VaderSentiment and normalizes the score between 1 to 10.
    (1 being most negative sentiment and 10 being most positive sentiment.)
- Updates the sentiment scores in the database and prints it.

## Run it

```bash
pip install -r requirements.txt
python GoodreadsReviewSentimentAnalyser.py
```

Live scraping is switched off because Goodreads rate-limits repeated requests. The script runs on 50 bundled sample reviews (10 each for five books) and prints every review with its sentiment score. The original scraper is kept in the script; to re-enable it, uncomment that block and set the book URL in the `requests.get(...)` call.

## Result:
In manual testing of a select number of books, I identified misrepresented star ratings for 20% of the analyzed books.

## Future Work:
- Increase the number of reviews used to analyze each book.
- Validate the VaderSentiment model for accuracy and take it's shortcomings into account for margin of error.
- Create a user friendly application for better accessibility.
