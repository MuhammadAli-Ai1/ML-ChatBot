import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import os
import sys
import subprocess

# Download necessary data for text processing
nltk.download('punkt', quiet=True)


#1 THE DATASET
knowledge_base = [
    # Greetings & Conversational
    "Hello there! How can I help you with Machine Learning today?",
    "I am a chatbot designed to answer questions about ML, AI, and Data Science.",
    "You can ask me about algorithms, definitions, or types.",
    "Goodbye! Happy coding!",

    #General Definitions
    "Machine Learning is a subset of AI that allows systems to learn from data without being explicitly programmed.",
    "Artificial Intelligence/AI is the broad field of creating smart machines capable of mimicking human intelligence.",
    "Data Science is a field that uses scientific methods, processes, algorithms, and systems to extract knowledge from data.",
    
    #Types of Learning
    "Supervised learning uses labeled data (input-output pairs) to train models.",
    "Unsupervised learning finds hidden patterns or structures in unlabeled data.",
    "Reinforcement learning is about an agent taking actions in an environment to maximize cumulative rewards.",
    "Semi-supervise learning falls between supervised and unsupervised learning, using a small amount of labeled data and a lot of unlabeled data.",

    #Key Algorithms
    "Linear Regression is a simple algorithm used for predicting a continuous numerical value.",
    "Logistic Regression is used for binary classification problems, like predicting 'Yes' or 'No'.",
    "A Decision Tree is a flowchart-like model used for both classification and regression.",
    "Random Forest is an ensemble method that creates many decision trees to improve prediction accuracy.",
    "K-Means Clustering is an unsupervised algorithm that groups data points into 'K' number of clusters.",
    "Neural Networks are a series of algorithms that mimic the operations of a human brain to recognize relationships in data.",
    "Deep Learning is a specialized subset of ML that uses multi-layered neural networks to solve complex problems.",
    "Support Vector Machines (SVM) are powerful supervised learning models used for classification and regression.",

    #Concepts & Terminology
    "Overfitting happens when a model learns the training data (and its noise) too well, performing poorly on new data.",
    "Underfitting occurs when a model is too simple to capture the underlying pattern of the data.",
    "Gradient Descent is an optimization algorithm used to minimize the loss function by iteratively moving towards the minimum.",
    "Features are the individual independent variables (inputs) that act as columns in your dataset.",
    "Labels are the dependent variables (targets) that you are trying to predict.",
    "Training Data is the subset of data used to fit the model.",
    "Testing Data is the subset of data kept hidden to evaluate how well the model generalizes.",
    
    #Metrics
    "Accuracy is the percentage of correct predictions made by the model.",
    "Precision tells us what proportion of positive identifications was actually correct.",
    "Recall (Sensitivity) measures the proportion of actual positives that were identified correctly.",
    "Confusion Matrix is a table used to describe the performance of a classification model.",

    #Applications
    "Natural Language Processing (NLP) is the branch of AI focused on the interaction between computers and human language.",
    "Computer Vision is a field of AI that enables computers to derive meaningful information from digital images and videos."
]


#THE BRAIN (The ML Logic)
def get_ml_response(user_query):
    # Combine query with knowledge base
    temp_list = knowledge_base + [user_query]
    
    # Transform text into a numerical matrix (TF-IDF)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(temp_list)
    
    # Calculate similarity b/W the query (last item) and the knowledge base
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    
    # Find the index of the best match words
    best_match_idx = similarity_scores.argsort()[0][-1]
    highest_score = similarity_scores[0][best_match_idx]

    #If the match is too weak, say we don't know
    if highest_score < 0.2:
        return "I'm not sure about that. Ask me about Supervised, Unsupervised, or Neural Networks!"
    
    return knowledge_base[best_match_idx]

# To Create THE STREAMLIT GUI
st.title("🎓 ML Basics Tutor") 
st.write("Ask me anything to expalin about Machine Learning fundamentals!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is Supervised Learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = get_ml_response(prompt)
    
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response })
#To auto run streamlit without cmd
if __name__ == "__main__" and not os.environ.get("STREAMLIT_SERVER_RUNNING"):
        os.environ["STREAMLIT_SERVER_RUNNING"] = "1"
        subprocess.Popen([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        os.path.abspath(__file__)
    ])
sys.exit()