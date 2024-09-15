## 🚀 Goal

The goal of this project is to create an educational and interactive Streamlit application that helps users, particularly those new to Natural Language Processing (NLP), understand and visualize key concepts in text processing.

## ✨ Features

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

## 🛠️ Built With

- [Streamlit](https://streamlit.io/) - The fastest way to build data apps
- [NLTK](https://www.nltk.org/) - Leading platform for building Python programs to work with human language data
- [Ollama](https://ollama.ai/) - Run large language models locally

## 🏁 Getting Started

### Prerequisites

- Python 3.7+
- pip
- Ollama (with the "llama2" model installed)

### Installation

1. Clone the repo:


```
git clone https://github.com/iRahulPandey/NLP_app.git

```

Navigate to the project directory:

```
cd NLP_app

```

Install required packages:

```
pip install -r requirements.txt

```

Download nltk data and copy it to a project folder:

```
python download_nltk_data.py

```

Run the Streamlit app:

```
streamlit run app.py

```

🖥️ Usage
- Enter a sentence in the text area.
- Watch as the app instantly displays various tokenization methods and POS tagging.
- Experiment with different sentences to see how the results change.
- Read the explanations to deepen your understanding of each concept.
