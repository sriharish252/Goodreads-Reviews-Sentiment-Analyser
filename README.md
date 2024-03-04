# Goodreads Reviews Sentiment Analyser

## Purpose:
Derive an aggregate sentiment score of a book to gain a more accurate and objective measure of reader satisfaction, attained through textual analysis.

## Stages of Execution:
- Scrapes reviews from the Goodreads website.
- Inserts initial scraped data into an SQLite database.
- Analyzes the reviews using VaderSentiment and normalizes the score between 1 to 10.
    (1 being most negative sentiment and 10 being most positive sentiment.)
- Updates the sentiment scores in the database and prints it.

## Prerequisites:
```
pip install requests  # HTTP library
pip install beautifulsoup4    # For pulling data out of HTML and XML files
pip install vaderSentiment    # For sentiment analysis of reviews
```

## Usage:
- Replace the URL in page = requests.get('https://www.goodreads.com/book/show/44581530-dead-astronauts') with the URL of the desired book on Goodreads.
- Run the script to scrape reviews and analyze sentiment.
- Sentiment score interpretation: 1 being most negative sentiment and 10 being most positive sentiment.

## Result:
In manual testing of a select number of books, I identified misrepresented star ratings for 20% of the analyzed books.

## Future Work:
- Increase the number of reviews used to analyze each book.
- Validate the VaderSentiment model for accuracy and take it's shortcomings into account for margin of error.
- Create a user friendly application for better accessibility.
