import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# LOAD DATA THROUGH PANDAS (read_csv)
def load_data():
    try:
        file = pd.read_csv("movie_info.csv")
        print("File loaded successfully")
    except ValueError:
        print("File did not load correctly. Please check file structure.")
    return file

# COMPUTE RECOMMENDATIONS
def compute_similarity(query, dataset, top_n):
    # BUILD VECTORS
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(dataset["Plot"])
    user_vector = vectorizer.transform([query])

    # COMPUTE COSINE SIMILARITY AND ASSIGN RECOMMENDATIONS TO VARIABLE
    similiarity_score = cosine_similarity(user_vector, tfidf_matrix).flatten()
    top_matches = similiarity_score.argsort()[-top_n:][::-1]
    recommendations = dataset.iloc[top_matches][["Title", "Plot"]]

    return recommendations

# MAIN ENTRY PT
def main():
    df = load_data()
    criteria_input = input("What content would you like?: ")
    match_input = input("How many recommendations would you like? (3-5 recommended): ")

    # CALL TO FUNCTION
    try:
        num_matches = int(match_input)
        recs = compute_similarity(criteria_input, df, num_matches)
        return recs
    except ValueError:
        print("ERROR: Invalid characters, please input an integer value for recommendations.")
    

    



if __name__ == "__main__":
    main()