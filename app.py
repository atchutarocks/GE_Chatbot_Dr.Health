import streamlit as st 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,AutoConfig
from transformers import pipeline
import torch
import base64
import textwrap
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma 
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
import chromadb

#model and tokenizer loading
device = torch.device('cpu')

# ---------
##Enter MODEL Name below
# --------
checkpoint = "google/flan-t5-small"   ##"MBZUAI/LaMini-T5-738M"
print(f"Checkpoint path: {checkpoint}")  # Add this line for debuggingtokenizer = AutoTokenizer.from_pretrained(checkpoint)
# base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    device_map=device,
    torch_dtype=torch.float32
)
@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 256,
        do_sample=True,
        temperature = 0.3,
        top_p = 0.95
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm
# ... [Other Imports] ...

def qa_llm(product_family):
    llm = llm_pipeline()
    
    # Set the persist_directory based on product family
    if product_family == "User":
        persist_dir = "db_user"
    elif product_family == "Installation":
        persist_dir = "db_installation"
    else:  # Others or any other category
        persist_dir = "db_others"
    
    embeddings = SentenceTransformerEmbeddings(model_name="multi-qa-MiniLM-L6-cos-v1")
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa

def process_answer(instruction, product_family):
    response = ''
    instruction = instruction
    qa = qa_llm(product_family) # Send product family to qa_llm
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer, generated_text


# List of available languages and their respective models
LANGUAGES = {
    "English": "google/flan-t5-small",
    "Arabic": "inception-mbzuai/jais-13b-chat",
    # Add more languages and their respective models here
}


def main():
    st.title("Generative Chatbot for GE HealthCare Customer Documentation Portal")
    
    with st.expander("About the App"):
        st.markdown(
        """
        Made by Team-Dr.Health \n
        Welcome to GE HealthCare's Documentation Assistant. Harnessing advanced Generative AI, this tool is designed to swiftly and accurately retrieve information from GEHC's vast repository of product manuals, data sheets, and operation guides. Whether you're a customer or a field engineer, our AI-powered chatbot is here to guide you through GE's medical technology documentation, supporting multiple languages to cater to our global user base.
        """
        )
    
    # Dropdown to select the language
    selected_language = st.selectbox("Select Language:", list(LANGUAGES.keys()))
    
    # Update the checkpoint variable based on the selected language
    global checkpoint
    checkpoint = LANGUAGES[selected_language]
    print(f"Checkpoint path: {checkpoint}")  # Add this line for debugging

   
 
    # Dropdown to select the product family
    product_family = st.selectbox("Select Product Family:", ["User", "Installation", "Others"])
    
    question = st.text_area("Enter your Question")
    if st.button("Ask"):
        st.info("Your Question: " + question)
        st.info("Dr.Health's Answer")
        answer, metadata = process_answer(question, product_family)  # Send product family to process_answer
        st.write(answer)
        st.write(metadata)

if __name__ == '__main__':
    main()
