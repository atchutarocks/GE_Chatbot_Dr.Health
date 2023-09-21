# GE_Chatbot_Dr.Health
Made by Team-Dr.Health \n
Welcome to GE HealthCare's Documentation Assistant. Harnessing advanced Generative AI, this tool is designed to swiftly and accurately retrieve information from GEHC's vast repository of product manuals, data sheets, and operation guides. Whether you're a customer or a field engineer, our AI-powered chatbot is here to guide you through GE's medical technology documentation, supporting multiple languages to cater to our global user base.
# Getting Started
Follow these steps to set up and run the project on your local machine.
# Installation
Clone the repository<br>
```
git clone https://github.com/atchutarocks/GE_Chatbot_Dr.Health.git 
```
Install all the requirements<br>
```
pip install -r requirements.txt
```
Create a directory/folder to store the PDF's of each section (i.e Service,User/Operation/Installation/Others)
Run the ingestion script to prepare the data <br>
```
python ingest.py 
```
The above script should create a new folder called db_<type_name> , based on the section chosen in persist_directory for ingest.py <br>
Start the chatbot application using Streamlit
```
streamlit run app.py
```
# Updating the LLM's being used 


