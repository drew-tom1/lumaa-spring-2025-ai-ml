import pandas as pd
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def initialize_nltk():
    """
    Function to download NLTK assets
    """
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('punkt_tab')

def load_data():
    """
    Function to load and return the dataset

    Parameters:
    None

    Returns:
    file (csv): Kaggle Dataset of movie info.
    """
    try:
        file = pd.read_csv("movie_info.csv")
        if "Plot" not in file.columns:
            raise KeyError("The 'Plot' column is missing in the dataset.")
        print("File loaded successfully")
        return file
    except (ValueError, KeyError) as e:
        print(f"Error loading file: {e}")
        return None


def text_preprocessing(text):
    """
    Preprocessing steps to reduce noise in the dataset entries.

    - Converts characters to lowercases
    - Removes punctuation
    - Tokenizes words
    - Removes stopwords
    - Applies lemmatization.

    Parameters:
    text (str): Input text to be processed

    Returns:
    Clean and preprocessed text (str).

    """
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer() 

    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return ' '.join(words)

def compute_similarity(query, dataset, top_n):
    """
    Function to compute similarity between user query and dataset entries.

    Parameters:
    query (str): User query of their criteria
    dataset (pandas DataFrame): CSV file of movie information.
    top_n (int): n number of recommendations wanted.

    Returns:
    recommendations (pandas DataFrame): n number of recommendations.
    """
    # APPLY PREPROCESSING TO DATA (USER INPUT AND DATASET)
    dataset["Processed_Plot"] = dataset["Plot"].apply(text_preprocessing)
    processed_query = text_preprocessing(query)

    # BUILD VECTORS
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(dataset["Processed_Plot"])
    user_vector = vectorizer.transform([processed_query])

    # COMPUTE COSINE SIMILARITY AND ASSIGN RECOMMENDATIONS TO VARIABLE
    similiarity_score = cosine_similarity(user_vector, tfidf_matrix).flatten()
    top_matches = similiarity_score.argsort()[-min(top_n, len(dataset)):][::-1]
    recommendations = dataset.iloc[top_matches][["Title", "Plot"]]

    return recommendations

# MAIN ENTRY PT
def main():
    initialize_nltk()
    df = load_data()
    criteria_input = input("What content would you like?: ")
    match_input = input("How many recommendations would you like? (3-5 recommended): ")

    # CALL TO FUNCTION
    try:
        num_matches = int(match_input)
        recs = compute_similarity(criteria_input, df, num_matches)
        print(recs) # OUTPUT FINAL RECOMMENDATIONS
    except ValueError:
        print("ERROR: Invalid characters, please input an integer value for recommendations.")
    

    



if __name__ == "__main__":
    main()