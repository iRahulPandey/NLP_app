import streamlit as st
from langchain_community.llms import Ollama
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import LatentDirichletAllocation  # Correct import added here
from sklearn.feature_extraction.text import CountVectorizer
import umap.umap_ as umap
import numpy as np
import spacy
from rake_nltk import Rake
from gensim.models import Word2Vec
from langdetect import detect
import re
import string

# Load spaCy model
nlp_spacy = spacy.load('en_core_web_sm')

# Initialize Ollama
ollama_llm = Ollama(model="llama3.1")

def ollama_generate(prompt, model):
    """Generates text using the Ollama model based on a given prompt."""
    try:
        response = model(prompt)
        return response
    except Exception as e:
        return f"An error occurred: {e}"
 
def apply_lowercase(text):
    return text.lower()

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return ' '.join(word for word in word_tokenize(text) if word.lower() not in stop_words)

def generate_summary(text, model):
    """Generates a summary of the given text."""
    prompt = f"""
                You are an expert summarizer. Your task is to create a concise and informative summary of the following text. 
                Focus on capturing the key points, main ideas, and important details. The summary should be clear, coherent, and approximately one-third the length of the original text.

                Text:
                {text}

                Summary:
                """
    return ollama_generate(prompt, model)

def identify_topics(text, model):
    """Identifies main topics from the text."""
    prompt = f"""
                You are an expert in topic modeling. Identify the main topics discussed in the following text. 
                For each topic, provide a brief description and list the relevant keywords associated with it. 
                Focus on extracting three to five distinct topics that best represent the overall content.

                Text:
                {text}

                Topics:
                """
    return ollama_generate(prompt, model)


def generate_explanation(text, model):
    """Generates an explanation for how the summary and topics were derived."""
    prompt = f"""
                You are an NLP expert explaining the process of summarization and topic extraction to a novice. 
                Provide a step-by-step explanation of how the summary and topics were derived from the following text. 
                Make sure to include the key elements considered in the summarization and the criteria used to identify the main topics.

                Text:
                {text}

                Explanation:
                """
    return ollama_generate(prompt, model)


def aspect_based_sentiment_analysis(text, model):
    """Performs aspect-based sentiment analysis on the text."""
    prompt = f"""
                You are a sentiment analysis expert. Analyze the following text to identify the main topics (e.g., product categories or entities) and their associated aspects (e.g., features, qualities, or attributes).
                For each topic, provide a list of aspects mentioned, the user's opinion about each aspect, and the overall sentiment (Positive, Negative, or Neutral) for the topic.
                Present the results in the following structured format:

                - **Topic**: [Topic Name]
                - **Aspect**: [Aspect Name]
                - **Opinion**: [User's opinion about the aspect]
                - **Sentiment**: [Positive/Negative/Neutral]

                Ensure that the output is grouped by topic and formatted clearly, like this example:

                - **Topic**: Headphones
                    - **Aspect**: Sound quality
                        - **Opinion**: Crystal clear
                        - **Sentiment**: Positive
                    - **Aspect**: Battery life
                        - **Opinion**: Impressive
                        - **Sentiment**: Positive
                    - **Aspect**: Comfort
                        - **Opinion**: Highly recommendable
                        - **Sentiment**: Positive
                    - **Overall Sentiment**: Positive

                Text:
                {text}

                Results:
                """
    return ollama_generate(prompt, model)


def perform_topic_modeling(text, n_topics=3):
    """Performs topic modeling on the text using LDA."""
    # Split the text into sentences to create a pseudo-corpus
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return {}, []

    # Vectorize the text
    vectorizer = CountVectorizer(
        stop_words='english',
        max_df=0.95,  # Adjust to ensure it doesn't conflict with min_df
        min_df=1      # Adjusted to ensure there is no conflict with max_df
    )
    try:
        X = vectorizer.fit_transform(sentences)
    except ValueError as e:
        st.error(f"An error occurred during topic modeling: {e}")
        return {}, []

    # Fit LDA model
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42
    )
    lda.fit(X)

    # Transform the data to topic space
    lda_output = lda.transform(X)

    # Determine the number of components for UMAP
    # Ensure n_components is less than the number of samples
    n_samples = lda_output.shape[0]
    n_components_umap = min(2, n_samples - 1)  # Ensure at least 2 components or less than the number of samples

    # Reduce dimensionality with UMAP
    umap_model = umap.UMAP(n_neighbors=min(15, n_samples - 1), n_components=n_components_umap, random_state=42)
    try:
        umap_embeddings = umap_model.fit_transform(lda_output)
    except ValueError as e:
        st.error(f"An error occurred during UMAP reduction: {e}")
        return {}, []

    # Prepare data for visualization
    topic_names = ["Topic " + str(i+1) for i in range(n_topics)]
    dominant_topic = np.argmax(lda_output, axis=1)
    data = {
        'x': umap_embeddings[:, 0],
        'y': umap_embeddings[:, 1],
        'sentence': sentences,
        'topic': [topic_names[i] for i in dominant_topic]
    }
    return data, topic_names


def perform_pos_tagging(text):
    """Performs part-of-speech tagging on the text."""
    doc = nlp_spacy(text)
    return [(token.text, token.pos_) for token in doc]

def perform_ner(text):
    """Performs named entity recognition on the text."""
    doc = nlp_spacy(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def extract_keywords(text):
    """Extracts keywords from the text using the RAKE algorithm."""
    rake_nltk_var = Rake()
    rake_nltk_var.extract_keywords_from_text(text)
    return rake_nltk_var.get_ranked_phrases()

def identify_stop_words(text):
    """Identifies stop words in the text."""
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return list(set([word for word in words if word.lower() in stop_words]))

def tokenize_and_lemmatize(text):
    """Tokenizes and lemmatizes the text."""
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('english')]
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens, lemmas

def generate_word_embeddings(text):
    """Generates word embeddings for the text."""
    sentences = sent_tokenize(text)
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
    filtered_sentences = [[word for word in sentence if word.isalpha() and word.lower() not in stopwords.words('english')] for sentence in tokenized_sentences]

    if len(filtered_sentences) < 2:
        return None

    model = Word2Vec(filtered_sentences, vector_size=50, min_count=1, workers=2)
    return model.wv

def extract_quotations(text):
    """Extracts quotations from the text."""
    quotations = re.findall(r'“([^”]+)”|\'([^\']+)\'|"([^"]+)"', text)
    return [quote for group in quotations for quote in group if quote]

def extract_facts(text, model):
    """Extracts facts from the text."""
    prompt = f"""
                You are an expert fact extractor. Your task is to extract key facts and figures from the following text. 
                Focus on identifying verifiable information such as statistics, dates, names, events, and other factual details. 
                Present the extracted facts in a bullet-point format for easy readability.

                Text:
                {text}

                Facts:
                - 
                """
    return ollama_generate(prompt, model)


def detect_language(text):
    """Detects the language of the given text."""
    try:
        return detect(text)
    except Exception:
        return "Could not detect language."

