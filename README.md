<img width="434" height="191" alt="Screenshot 2026-10-09 at 3 13 22 PM" src="https://github.com/user-attachments/assets/e44bd89e-2d9f-468d-b892-4bc568a5cb9a" />
<img width="474" height="180" alt="Screenshot 2026-10-09 at 3 13 01 PM" src="https://github.com/user-attachments/assets/92797e83-3760-40df-9fd4-1872bbac9db6" />
# Movie Review Sentiment Analysis

Give it a movie review and it tells you whether the person liked the movie or not.

That's the whole idea. It's a small classic NLP project: clean up 50,000 IMDB reviews, turn the words into numbers, and train a couple of Naive Bayes models to tell positive reviews from negative ones.

The best model gets about **84% accuracy** on reviews it has never seen. Not state of the art, but pretty good for a model that trains in a few seconds and has no idea what a "movie" is.

---

## What's in the repo

There isn't much here, and that's on purpose.

```
SENTIMENT-ANALYSIS/
├── IMDB-Dataset.csv                      # 50k labelled movie reviews (~66 MB)
├── MoviewReviewSentimentOriginal.ipynb   # the whole project lives here
└── README.md                             # you're reading it
```

All the work happens in the notebook. There are no separate scripts or packages. Open it and run it from top to bottom.

When you run it, the notebook also writes two files into the same folder:

- `model1.pkl`: the trained Bernoulli Naive Bayes model
- `bow.pkl`: the vocabulary (word → column index) used to build the features

They aren't committed. You'll get them the first time you run the notebook.

---

## The dataset

`IMDB-Dataset.csv` has two columns:

| column      | what it holds                          |
|-------------|----------------------------------------|
| `review`    | the raw review text, HTML tags and all |
| `sentiment` | `positive` or `negative`               |

There are 50,000 rows, split exactly in half: 25,000 positive and 25,000 negative. Because the classes are balanced, plain accuracy is a fair number to report. You don't need to worry about a model that just guesses "positive" every time and still looks good.

---

## How it works

Here's the pipeline, step by step. If you read the notebook alongside this, the cells follow the same order.

### 1. Load and look

Read the CSV with pandas, check the shape, and confirm the 50/50 split. Then map the labels to numbers:

- `positive` → `1`
- `negative` → `0`

### 2. Clean the text

Raw reviews are messy. They have `<br />` tags all over the place, plus punctuation, capital letters, and lots of filler words. The cleanup runs in five steps, each one a small function:

1. **`clean_text`**: strips HTML tags with a regex
2. **`is_special`**: replaces anything that isn't a letter or digit with a space
3. **`to_lower`**: lowercases everything
4. **`rem_stopwords`**: tokenizes with NLTK and drops English stopwords ("the", "is", "and", and so on)
5. **`stem_text`**: runs NLTK's Snowball stemmer, so "watching", "watched" and "watches" all become "watch"

To see what that does, here's the start of the very first review, before:

> One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. They are right, as this is exactly what happened with me.<br /><br />The first thing that struck me...

And after:

> one review mention watch 1 oz episod hook right exact happen first thing struck...

It's ugly to read, but the model doesn't care about that. It only needs the signal.

### 3. Bag of words

The cleaned text goes through scikit-learn's `CountVectorizer` with `max_features=1000`. In plain terms, it picks the 1,000 most common (stemmed) words in the corpus and represents every review as a row of counts for those words.

So the full dataset becomes a `50,000 × 1,000` matrix.

### 4. Train / test split

It's an 80/20 split with `random_state=9`, so the results are reproducible:

- train: 40,000 reviews
- test: 10,000 reviews

### 5. Train two models

Two Naive Bayes models are trained and compared:

| model                | test accuracy |
|----------------------|---------------|
| Gaussian NB          | 78.4%         |
| **Bernoulli NB**     | **83.9%**     |

Bernoulli wins pretty clearly. That makes sense: it only cares about whether a word *shows up* in a review, not how many times. For sentiment, a single "awful" or "brilliant" tends to matter more than how often it gets repeated.

The Bernoulli model is the one that gets pickled.

### 6. Try it on new reviews

At the end of the notebook there are two sanity checks. Each review goes through the same five cleaning steps, then gets turned into features with the *same* vectorizer the model was trained on:

```python
inp = counvec.transform([f5]).toarray()
y_pred = bernou.predict(inp)
```

That second part matters. If the new review isn't mapped onto the exact same 1,000 words, the model is basically reading gibberish.

The results:

- a brutal one-star rant about an Avengers movie → predicted **0 (negative)** ✔
- the glowing review of *Oz* from the dataset → predicted **1 (positive)** ✔

<!--
SCREENSHOT PLACEHOLDER
Replace the line below with a screenshot of the notebook cell that prints:
  Gaussian Model =  0.7843
  Bernoulli Model =  0.8386
Save it as e.g. images/accuracy.png and update the path.
-->
<img width="427" height="183" alt="Screenshot 2026-10-09 at 3 13 47 PM" src="https://github.com/user-attachments/assets/fb73b62c-f79c-4a0c-b8a1-c9ed0abb8924" />

---

## Running it yourself

You'll need Python 3 (it was built on 3.13) and Jupyter.

**1. Clone it**

```bash
git clone https://github.com/Karan1223k/SENTIMENT-ANALYSIS.git
cd SENTIMENT-ANALYSIS
```

**2. Install the libraries**

```bash
pip install numpy pandas nltk scikit-learn jupyter
```

**3. Open the notebook**

```bash
jupyter notebook MoviewReviewSentimentOriginal.ipynb
```

Then just run all cells.

A couple of heads-ups:

- The notebook downloads the NLTK `stopwords` and `punkt` data on its own the first time. If you're on a newer NLTK version and tokenizing throws an error about `punkt_tab`, run `nltk.download('punkt_tab')` once and you're good.
- The cleaning steps go through all 50k reviews one by one, so they take a minute or two. Stemming is the slowest. Give it time, it hasn't frozen.

---

## Known rough edges

I'd rather be upfront about these than have someone find them the hard way.

- **The cleaning pipeline isn't packaged.** If you load `model1.pkl` somewhere else, you have to re-run the exact same five cleaning steps before predicting, or the words won't line up with the vocabulary.
- **Only 1,000 features.** Bumping `max_features` or switching to TF-IDF would very likely push accuracy up a few points.

---

## Where it could go next

Some ideas, if I (or you) come back to this:

- swap `CountVectorizer` for `TfidfVectorizer` and add bigrams
- try Logistic Regression or a linear SVM, both usually beat Naive Bayes on this dataset
- bundle the cleaning and the model into a single scikit-learn `Pipeline` so it's one `.pkl` and one `.predict()`
- put a tiny Streamlit or Flask front end on it so you can paste a review and get an answer

---

## Built with

- **pandas / NumPy**: data handling
- **NLTK**: tokenizing, stopwords, stemming
- **scikit-learn**: bag of words, train/test split, Naive Bayes, metrics
- **Jupyter**: where all of it runs

The dataset is the well-known IMDB 50K movie reviews set (Maas et al., 2011), available on Kaggle.
