import cohere
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import Cohere as LangchainCohere
import streamlit as st

# Initialize Cohere LLM via LangChain
cohere_llm = LangchainCohere(
    model="command",  # Specify the Cohere model
    cohere_api_key="UvQcnNHSF42oPDGTWv6P9OpGrOCMb9lPgKOjxj3m",  # Replace with your Cohere API key
    temperature=0.3,  # Adjust for creativity
    max_tokens=100  # Limit the length of the summary
)

# Define a Prompt Template for Summarization
prompt_template = PromptTemplate(
    input_variables=["text"],  # Dynamic input for the text to summarize
    template="Summarize the following text:\n\n{text}\n\nSummary:"
)

# Combine LLM with the Prompt into a LangChain
summarization_chain = LLMChain(
    llm=cohere_llm,
    prompt=prompt_template
)

# Streamlit UI
st.title("Text Summarization with Cohere and LangChain")

# Input Section
st.write("Enter text below to summarize it:")
input_text = st.text_area("Text Input", height=200)

# Button to Generate Summary
if st.button("Generate Summary"):
    if input_text.strip():
        with st.spinner("Generating summary..."):
            # Generate the summary
            summary = summarization_chain.run(text=input_text)
        st.success("Summary generated!")
        st.subheader("Summary")
        st.write(summary)
    else:
        st.error("Please enter some text to summarize.")
