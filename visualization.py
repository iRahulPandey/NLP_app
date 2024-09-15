import streamlit as st
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.resources import CDN
from bokeh.embed import file_html
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import plotly.express as px

def generate_word_cloud(text):
    """Generates a word cloud visualization from the text."""
    stopwords_set = set(STOPWORDS)
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords_set).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

def plot_interactive_topics(data, topic_names):
    """Creates an interactive topic visualization using Bokeh."""
    if not data:
        st.write("Not enough data for topic modeling.")
        return
    source = ColumnDataSource(data)

    # Create a color palette
    from bokeh.palettes import Category10
    num_topics = len(topic_names)
    colors = Category10[10]  # Category10 provides up to 10 colors
    if num_topics > 10:
        from bokeh.palettes import viridis
        colors = viridis(num_topics)
    topic_colors = {topic: colors[i % len(colors)] for i, topic in enumerate(topic_names)}
    data['color'] = [topic_colors[topic] for topic in data['topic']]
    source.data = data

    # Create the plot
    TOOLTIPS = [
        ("Sentence", "@sentence"),
        ("Topic", "@topic")
    ]

    p = figure(title="Interactive Topic Visualization",
               plot_width=700, plot_height=600,
               tools=("pan,wheel_zoom,reset"),
               tooltips=TOOLTIPS)

    p.scatter('x', 'y', color='color', source=source, legend_field='topic', alpha=0.6, size=10)
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"

    # Convert the Bokeh plot to HTML
    html = file_html(p, CDN, "Topic Visualization")
    st.components.v1.html(html, height=600)

def visualize_word_embeddings(word_vectors):
    """Visualizes word embeddings using t-SNE and Plotly."""
    if not word_vectors:
        st.write("Not enough data for word embeddings.")
        return
    words = word_vectors.index_to_key
    embeddings = np.array([word_vectors[word] for word in words])  # Convert to a NumPy array
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    df_embeddings = pd.DataFrame({
        'word': words,
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1]
    })
    fig = px.scatter(df_embeddings, x='x', y='y', text='word')
    fig.update_traces(textposition='top center')
    fig.update_layout(title='Word Embeddings Visualization (t-SNE)', xaxis_title='Dimension 1', yaxis_title='Dimension 2')
    st.plotly_chart(fig)
