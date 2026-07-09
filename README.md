# movie-recommendation-system
# 🎬 Content-Based Movie Recommendation System

A Machine Learning project that recommends movies similar to a selected movie using **Content-Based Filtering**. The recommendation engine analyzes movie metadata such as genres, keywords, cast, crew, and overview to suggest relevant movies.

---

## 📌 Project Overview

This project builds an intelligent movie recommendation system using:

* Natural Language Processing (NLP)
* Feature Engineering
* TF-IDF Vectorization
* Cosine Similarity

The system recommends movies based on their content rather than user ratings.

---

## 🚀 Features

* Top-10 movie recommendations
* Content-based filtering approach
* NLP preprocessing and stemming
* TF-IDF vectorization
* Cosine similarity-based recommendations
* Developed entirely in **Google Colab**

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* NLTK
* Pickle
* Google Colab

---

## 📂 Dataset

Dataset: **TMDB 5000 Movie Dataset**

Files used:

```text
tmdb_5000_movies.csv
tmdb_5000_credits.csv
```

The dataset contains:

* Movie Title
* Genres
* Keywords
* Cast Information
* Crew Information
* Movie Overview
* Popularity and Ratings

---

## ⚙️ Machine Learning Pipeline

```text
Data Collection
       ↓
Data Preprocessing
       ↓
Feature Engineering
       ↓
Text Preprocessing
       ↓
TF-IDF Vectorization
       ↓
Cosine Similarity
       ↓
Recommendation Engine
```

---

## 📊 Data Preprocessing

* Merged movies and credits datasets
* Removed missing values and duplicates
* Extracted:

  * Genres
  * Keywords
  * Top 3 Cast Members
  * Director
* Created a combined `tags` feature

---

## 🧠 Feature Engineering

The following features are combined into a single column:

```python
tags = overview + genres + keywords + cast + crew
```

Example:

```text
action adventure future samworthington zoesaldana jamescameron
```

---

## 🤖 Model Building

### Text Vectorization

```python
TfidfVectorizer(max_features=5000,
                stop_words='english')
```

### Similarity Metric

```python
cosine_similarity(vectors)
```

---

## 🚀 Run in Google Colab

### Install Dependencies

```python
!pip install pandas numpy scikit-learn nltk
```

### Download NLTK Resources

```python
import nltk
nltk.download('stopwords')
```

### Upload Dataset Files

```text
tmdb_5000_movies.csv
tmdb_5000_credits.csv
```

### Execute the Notebook Cells

Run all cells in sequence and use:

```python
recommend('Avatar')
```

---

## 🎯 Example Output

```text
Aliens
Titan A.E.
Guardians of the Galaxy
Star Trek
John Carter
The Matrix
Moonraker
Battle: Los Angeles
Wing Commander
Star Trek Into Darkness
```

---

## 📈 Skills Demonstrated

* Data Cleaning and Preprocessing
* Feature Engineering
* Natural Language Processing
* Recommendation Systems
* Similarity Metrics
* Machine Learning Pipeline Development
* Model Serialization

---

## 🔮 Future Enhancements

* Hybrid Recommendation System
* Fuzzy Search for Movie Names
* Movie Poster Integration
* Deep Learning-based Recommendation
* Web Application Deployment

---

## 👨‍💻 Author

**V. Muthuvel**
B.Tech – Artificial Intelligence and Data Science
NPR College of Engineering and Technology

---

## ⭐ If you found this project useful, please consider giving it a star on GitHub!
