import csv
import sqlite3
from sqlite3 import Error
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

#SQLite
#Create Table Format
create_table = """ CREATE TABLE IF NOT EXISTS reviews (
    bid integer NOT NULL,
    bname text NOT NULL,
    rno integer NOT NULL,
    review text,
    score int
); """

#Create Connection
conn = None
try:
    conn = sqlite3.connect(':memory:')    #(':memory:') To create DB in RAM  #(r"G:\CLG\Internship\GoodreadsReviewSentiment.db")
    print("Database Created")
except Error as e:
    print(e)

#Create Table
try:
    c = conn.cursor()
    c.execute(create_table)
    print("Table Created")
except Error as e:
    print(e)

c.execute('DELETE FROM reviews')
conn.commit()


#Insert data into Table
with open('data/sample_reviews.csv', encoding='utf-8') as f:
    rev = [x for row in csv.DictReader(f) for x in (row['book'], row['review'])]

ctr = 0
c = conn.cursor()
for i in range(5):
    for j in range(10):
        c.execute('''INSERT INTO reviews(bid,bname,rno,review,score) VALUES(?,?,?,?,?)''',(i+1,rev[ctr],j+1,rev[ctr+1],'None'))
        ctr=ctr+2
print("Rows Inserted")
conn.commit()   #To make changes permanent by stating that this is an end of the Transaction

def sentiment_analyzer_scores(sentence):
    sentiment_score = analyser.polarity_scores(sentence)
    sentiment_score = sentiment_score['compound']*(9/2)+(11/2) #Normalizing to 1 to 10
    return sentiment_score

print("\nInitial Values\n")
c1 = conn.cursor()
c.execute('''SELECT * FROM reviews''')
for row in c:
    # row[0] returns the first column in the query (bid), row[1] returns bname column.
    print(('{0} : {1}, {2} : {3}, {4}'.format(row[0], row[1], row[2], row[3], row[4])).encode('utf8'))
    sentiment_score = sentiment_analyzer_scores(row[3])
    c1.execute('''UPDATE reviews SET score=? WHERE bid=? AND rno=?''',(sentiment_score,row[0],row[2]))

#Final Output
print("\n\nOUTPUT:\n")
c.execute('''SELECT * FROM reviews''')
for row in c:
    # row[0] returns the first column in the query (bid), row[1] returns bname column.
    print(('{0} : {1}, {2} : {3}, {4}'.format(row[0], row[1], row[2], row[3], row[4])).encode('utf8'))

conn.commit()
conn.close()
