import subprocess
import streamlit as st

def get_ollama_models():
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        models = [line.split()[0] for line in result.stdout.strip().split('\n')[1:]]
        return models
    except Exception as e:
        st.error(f"Error fetching Ollama models: {str(e)}")
        return ["mixtral", "llama3.1"]
        