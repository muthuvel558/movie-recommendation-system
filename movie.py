#libraries
import pandas as pd
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#data
data = {
    "title": [
        "Baahubali",
        "Vikram",
        "Asuran",
        "96",
        "Enthiran",
        "Kaithi",
        "Jai Bhim",
        "Alaipayuthey",
        "Mersal",
        "Super Deluxe"
    ],
    "description": [
        "A legendary warrior fights to reclaim his kingdom",
        "An undercover agent battles a powerful drug cartel",
        "A farmer protects his family from violent enemies",
        "Two childhood lovers reunite after many years",
        "A scientist creates a humanoid robot with emotions",
        "A prisoner helps police rescue kidnapped officers",
        "A lawyer fights for justice for oppressed tribal people",
        "A romantic story about love and marriage",
        "A doctor seeks revenge against corruption",
        "Multiple lives intersect in an unusual dramatic narrative"
    ],
    "genre": [
        "Epic Action",
        "Action Thriller",
        "Action Drama",
        "Romance Drama",
        "Sci-Fi Action",
        "Action Thriller",
        "Legal Drama",
        "Romance",
        "Action Thriller",
        "Drama Thriller"
    ]
}   

movies =pd.DataFrame(data)
movies["tags"] = movies["description"] + "" "" +movies["genre"]

# FEATURE ENGINEERING
vectorizer = TfidfVectorizer(stop_words="english")
vectors = vectorizer.fit_transform(movies["tags"])
similarity = cosine_similarity(vectors)

 # RECOMMENDATION FUNCTION
def recommend(movie_title):
    index = movies[movies["title"] == movie_title].index[0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
    return [movies.iloc[i[0]]["title"] for i in scores]

# GRADIO UI FUNCTION
def recommend_ui(movie):
    return recommend(movie)

# BUILD UI
app = gr.Interface(
    fn = recommend_ui,
    inputs = gr.Dropdown(
        choices=movies["title"].tolist(),
        label = "select movie"
    ),
    outputs =gr.List(label  = "recomannded system"),
    title="Movie Recommendation System",
    description="content and collabrative based movie recommandtion system"
)


app.launch()
