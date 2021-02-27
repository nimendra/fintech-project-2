## What we are going to do:  
  
Analayis of influence of "covid 19 vaccine" tweets and news articles on *US Stock* market. 
We'd focus on individual stocks, rather than indices. User Input: Search word, Stock Ticker, timeframe.
Decisions:
Every 15 mins - we focus on a someone seeing a tweet and reacting  
Let's group all tweets within that 15 mins in to a block of text
  


__For the presentation:__  
 'covid19vaccine',  
 '[APPL]',  
 '2020-09-01'-'2021-02-01'
    
  
**Wed 24th:**  
* Decide on what datasets
* Which columns
* We can use Alpaca API
* Get the framework in place, that can handle 15, 10 mins etc.

Action Item:
1. : Compound tweets to ever 15 mins, or look at the Kaggle, build the NLP
2. : News/ Reuters/ Bloomberg
3. : @Sam stock ticker 1
4. : stock ticker 2
5. : stock ticker 3




| Time       | Column 2     | Column 3     |
| :------------- | :----------: | -----------: |
|  2020-09-01 01:00 | Long tweet text   | long news text    |
|  2020-09-01 01:15 | Long tweet text   | long news text    |
|  2020-09-01 01:30 | Long tweet text   | long news text    |
|  2020-09-01 01:45 | Long tweet text   | long news text    |

| Time       | APPL     | AMZN     |
| :------------- | :----------: | -----------: |
|  2020-09-01 01:00 | 125.82   | 3194.50    |
|  2020-09-01 01:15 | 126.93   | 3195.40    |
|  2020-09-01 01:30 | 127.74   | 3196.30    |
|  2020-09-01 01:45 | 128.65   | 3197.20    |


**Sat 27th:**  
Decide on algorithms

1. Sam: Algorithm 1
2. Paul: Algorithm 2
3. Okena: Algorithm 3
4. Raj: Algorithm 4
5. Nim: Algorithm 5

**Mon 1st March:**

**Wed 3rd March:**

**Sat 6th:** Presentation

## Datasets:

NLPDF : `[date, text]`
StrockDF : `[date , closing]`


```
To https://github.com/nimendra/fintech-project-2.git
   e5da59b..ed01bb8  master -> master
(p37env) ➜  Project_2 git:(master) ✗ git checkout -b nim 
M       tweet.ipynb
Switched to a new branch 'nim'
(p37env) ➜  Project_2 git:(nim) ✗ git checkout master 
M       tweet.ipynb
Switched to branch 'master'
(p37env) ➜  Project_2 git:(master) ✗ git pull origin master
From https://github.com/nimendra/fintech-project-2
 * branch            master     -> FETCH_HEAD
Already up to date.
(p37env) ➜  Project_2 git:(master) ✗ git checkout nim
M       tweet.ipynb
Switched to branch 'nim'
(p37env) ➜  Project_2 git:(nim) ✗              

```

TODO:
'user_location'