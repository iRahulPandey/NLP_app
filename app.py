import streamlit as st
from langchain_community.llms import Ollama
import yaml
import pandas as pd
import spacy
from spacy import displacy
from text_processing import (
    generate_summary, identify_topics, generate_explanation,
    aspect_based_sentiment_analysis, perform_topic_modeling,
    perform_pos_tagging, perform_ner, extract_keywords,
    identify_stop_words, tokenize_and_lemmatize, generate_word_embeddings,
    extract_quotations, extract_facts, detect_language, apply_lowercase, remove_punctuation, remove_stopwords
)
from visualization import (
    plot_interactive_topics, generate_word_cloud, visualize_word_embeddings
)

from model_utils import get_ollama_models

# Load spaCy model
nlp_spacy = spacy.load('en_core_web_sm')

# Initialize Streamlit app
st.set_page_config(page_title="Local NLP Analysis App", layout="wide")
st.title("Natural Language Processing (NLP) App")

st.write("""Explore the power of NLP with this app! Analyze text for summaries, topics, sentiment, keywords, entities, and more. Discover insights and learn how these techniques work.""")

# Load configuration file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize session state variables if they do not exist
if "analyze_button_clicked" not in st.session_state:
    st.session_state.analyze_button_clicked = False
if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = {}
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Function to reset analysis state

def reset_analysis():
    st.session_state.analyze_button_clicked = False
    st.session_state.analysis_data = {}
    
# Dropdown to select Ollama model    
st.markdown("### Select Model")
ollama_models = get_ollama_models()
selected_model = st.selectbox("Choose Ollama Model", ollama_models, index=0, on_change=reset_analysis)

# Create Ollama model instance
ollama_llm = Ollama(model=selected_model)

# Sidebar with additional features
st.sidebar.header("Settings")

# Dropdown to select example text
selected_example = st.sidebar.selectbox("Select Example Text", list(config["example_texts"].keys()))

# Load selected example text
if st.sidebar.button("Use Selected Example Text"):
    st.session_state.user_input = config["example_texts"][selected_example]
    
# Main text input area

user_input = st.text_area("Enter the text you want to analyze:", height=200, value=st.session_state.user_input)

# Minimum length requirement
MIN_TEXT_LENGTH = 500  # Set minimum number of words required
MIN_SENTENCES = 10  # Minimum number of sentences required for specific tasks

# Analysis Button
if st.button("Analyze"):
    if len(user_input.split()) < MIN_TEXT_LENGTH:
        st.error(f"Please enter at least {MIN_TEXT_LENGTH} words for meaningful analysis.")
    elif user_input.strip() == "":
        st.error("Please enter some text to analyze.")
    else:
        st.session_state.analyze_button_clicked = True
        st.session_state.user_input = user_input  # Update session state with the latest input
        with st.spinner("Processing..."):
            # Perform analysis and store results in session state
            analysis_data = st.session_state.analysis_data
            analysis_data['selected_model'] = selected_model
            analysis_data['lowercase_text'] = apply_lowercase(user_input)
            analysis_data['punctuation_removed_text'] = remove_punctuation(analysis_data['lowercase_text'])
            analysis_data['stopwords_removed_text'] = remove_stopwords(analysis_data['punctuation_removed_text'])
            analysis_data['summary'] = generate_summary(user_input, ollama_llm)
            analysis_data['topics'] = identify_topics(user_input, ollama_llm)
            analysis_data['explanation'] = generate_explanation(user_input, ollama_llm)
            analysis_data['aspect_sentiment'] = aspect_based_sentiment_analysis(user_input, ollama_llm)
            
            # Perform Topic Modeling only if minimum sentence requirement is met
            sentences = user_input.split('.')
            if len(sentences) >= MIN_SENTENCES:
                analysis_data['topic_data'], analysis_data['topic_names'] = perform_topic_modeling(user_input)
            else:
                analysis_data['topic_data'], analysis_data['topic_names'] = None, None

            analysis_data['pos_tags'] = perform_pos_tagging(user_input)
            analysis_data['entities'] = perform_ner(user_input)
            analysis_data['keywords'] = extract_keywords(user_input)
            analysis_data['stop_words'] = identify_stop_words(user_input)
            analysis_data['tokens'], analysis_data['lemmas'] = tokenize_and_lemmatize(user_input)
            analysis_data['word_vectors'] = generate_word_embeddings(user_input)
            analysis_data['quotations'] = extract_quotations(user_input)
            analysis_data['facts'] = extract_facts(user_input, ollama_llm)
            analysis_data['language'] = detect_language(user_input)

# Display Analysis Results
if st.session_state.analyze_button_clicked:
    tabs = st.tabs([
        "Preprocessed Steps", "Summary", "Topics", "Interactive Topic Visualization", "Word Cloud",
        "Aspect-Based Sentiment Analysis", "POS Tagging", "NER", "Keywords & Stop Words",
        "Tokens & Lemmas", "Word Embeddings", "Quotations",
        "Fact Extraction", "Language Detection", "Model Explanation", "NLP Glossary"
    ])

    with tabs[0]:
        st.subheader("Preprocessing Steps")
        with st.expander("Learn More about Preprocessing"):
            st.write("""
            **Preprocessing** involves various steps to clean and prepare text for analysis. Here, we demonstrate each step's effect on your input text.
            """)
        st.write("### Original Text:")
        st.write(st.session_state.user_input)
        st.write("### Lowercased Text:")
        st.write(st.session_state.analysis_data['lowercase_text'])
        st.write("### Punctuation Removed:")
        st.write(st.session_state.analysis_data['punctuation_removed_text'])
        st.write("### Stopwords Removed:")
        st.write(st.session_state.analysis_data['stopwords_removed_text'])

    with tabs[1]:
        st.subheader("Summary")
        with st.expander("Learn More about Summarization"):
            st.write("""
            **Summarization** involves extracting the most important information from the text to provide a concise representation.
            This app uses the Ollama model to generate the summary.
            """) 
        st.write(f"**Model Used:** {st.session_state.analysis_data['selected_model']}")
        st.write(st.session_state.analysis_data['summary'])

    with tabs[2]:
        st.subheader("Topics (Model-Generated)")
        if st.session_state.analysis_data['topic_data']:
            with st.expander("Learn More about Topic Extraction"):
                st.write("""
                **Topic Extraction** identifies the main themes or topics present in the text. 
                """)
            st.write(f"**Model Used:** {st.session_state.analysis_data['selected_model']}")    
            st.write(st.session_state.analysis_data['topics'])
        else:
            st.write(f"Text is too short for topic modeling. Please enter at least {MIN_SENTENCES} sentences.")

    with tabs[3]:
        st.subheader("Interactive Topic Visualization")
        if st.session_state.analysis_data['topic_data']:
            with st.expander("Learn More about Topic Visualization"):
                st.write("""
                **Interactive Topic Visualization** uses UMAP to project high-dimensional data into a lower-dimensional space, allowing for better visualization of the identified topics.
                """)
            plot_interactive_topics(st.session_state.analysis_data['topic_data'], st.session_state.analysis_data['topic_names'])
        else:
            st.write(f"Text is too short for topic visualization. Please enter at least {MIN_SENTENCES} sentences.")

    with tabs[4]:
        st.subheader("Word Cloud")
        with st.expander("Learn More about Word Cloud"):
            st.write("""
            **Word Cloud** visualizes the most frequent words in the text. The size of each word in the cloud indicates its frequency or importance.
            """)
        generate_word_cloud(user_input)

    with tabs[5]:
        st.subheader("Aspect-Based Sentiment Analysis")
        with st.expander("Learn More about Aspect-Based Sentiment Analysis"):
            st.write("""
            **Aspect-Based Sentiment Analysis** breaks down text into different components (aspects) and determines the sentiment (positive, negative, or neutral) associated with each aspect.
            """)
        st.write(f"**Model Used:** {st.session_state.analysis_data['selected_model']}")    
        st.markdown(st.session_state.analysis_data['aspect_sentiment'])

    with tabs[6]:
        st.subheader("Part-of-Speech (POS) Tagging")
        with st.expander("Learn More about Part-of-Speech (POS) Tagging"):
            st.write("""
            **Part-of-Speech (POS) Tagging** involves assigning a part of speech to each word in the text, such as noun, verb, adjective, etc.
            """)
        st.dataframe(pd.DataFrame(st.session_state.analysis_data['pos_tags'], columns=['Token', 'POS Tag']))

    with tabs[7]:
        st.subheader("Named Entity Recognition (NER)")
        with st.expander("Learn More about Named Entity Recognition (NER)"):
            st.write("""
            **Named Entity Recognition (NER)** identifies and classifies named entities in text into predefined categories such as person names, organizations, locations, etc.
            """)
        st.dataframe(pd.DataFrame(st.session_state.analysis_data['entities'], columns=['Entity', 'Label']))
        # NER Visualization
        html = displacy.render(nlp_spacy(user_input), style="ent")
        st.markdown(html, unsafe_allow_html=True)

    with tabs[8]:
        st.subheader("Keyword Extraction")
        with st.expander("Learn More about Keyword Extraction"):
            st.write("""
            **Keyword Extraction** identifies the most significant words or phrases in the text.
            """)
        st.write(st.session_state.analysis_data['keywords'])
        st.subheader("Stop Words Found")
        with st.expander("Learn More about Stop Words"):
            st.write("""
            **Stop Words** are common words that are usually filtered out in text processing.
            """)
        st.write(st.session_state.analysis_data['stop_words'])

    with tabs[9]:
        st.subheader("Tokenization and Lemmatization")
        with st.expander("Learn More about Tokenization and Lemmatization"):
            st.write("""
            **Tokenization** breaks the text into individual words or tokens. **Lemmatization** reduces words to their base or dictionary form.
            """)
        token_lemma_df = pd.DataFrame({'Token': st.session_state.analysis_data['tokens'], 'Lemma': st.session_state.analysis_data['lemmas']})
        st.dataframe(token_lemma_df)

    with tabs[10]:
        st.subheader("Word Embeddings Visualization")
        with st.expander("Learn More about Word Embeddings Visualization"):
            st.write("""
            **Word Embeddings** are a technique in natural language processing that represent words as vectors of numbers, capturing the meaning and relationships between words.
            """)
        if st.session_state.analysis_data['word_vectors']:
            visualize_word_embeddings(st.session_state.analysis_data['word_vectors'])
        else:
            st.write("Text is too short for word embeddings visualization.")

    with tabs[11]:
        st.subheader("Quotation Extraction")
        with st.expander("Learn More about Quotation Extraction"):
            st.write("""
            **Quotation Extraction** extracts direct quotations or phrases from the text.
            """)
        st.write("Extracted quotations from the text:")
        for quote in st.session_state.analysis_data['quotations']:
            st.write(f"- {quote}")

    with tabs[12]:
        st.subheader("Fact Extraction")
        with st.expander("Learn More about Fact Extraction"):
            st.write("""
            **Fact Extraction** identifies and extracts factual information from the text.
            """)
        st.write(f"**Model Used:** {st.session_state.analysis_data['selected_model']}")
        st.write("Key facts and figures extracted from the text:")
        st.markdown(st.session_state.analysis_data['facts'])

    with tabs[13]:
        st.subheader("Language Detection")
        with st.expander("Learn More about Language Detection"):
            st.write("""
            **Language Detection** identifies the language of the input text.
            """)
        st.write(f"The detected language of the text is: **{st.session_state.analysis_data['language']}**")

    with tabs[14]:
        st.subheader("Model Explanation")
        with st.expander("Learn More about the Model"):
            st.write("""
            **Model Explanation** provides insights into how different NLP models and techniques are applied to the input text in this app.
            """)
        st.write(f"**Model Used:** {st.session_state.analysis_data['selected_model']}")    
        st.write(st.session_state.analysis_data['explanation'])

    with tabs[15]:
        st.subheader("NLP Glossary")
        with st.expander("Learn More about NLP Concepts"):
            st.write("""
            - **Lemmatization**: Reducing words to their base or root form.
            - **Word Embedding**: Representing words in a numerical vector space to capture semantic relationships.
            - **POS Tagging**: Assigning part of speech tags to each word in a text.
            - **NER**: Identifying named entities like people, organizations, locations, etc., in a text.
            - **Topic Modeling**: Identifying hidden topics within a set of documents.
            - **Sentiment Analysis**: Determining the sentiment (positive, negative, or neutral) expressed in a text.
            - **Tokenization**: Breaking down text into individual words or tokens.
            - **Stop Words**: Commonly used words (like "the", "is", "in") that are usually removed during text processing because they carry less meaning.
            - **Fact Extraction**: Identifying and extracting factual information from the text.
            - **Quotation Extraction**: Extracting direct quotations or phrases from the text.
            """)
