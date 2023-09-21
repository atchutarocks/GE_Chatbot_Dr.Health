from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma 
import os 
import chromadb
import time

### Give the corresponding directory's name below - could be db_installation,db_user,db_all
persist_directory="db_installation"

def main():
    all_texts = []  # to store texts from all PDFs
    start = time.time()
    ##print("Here near start")
    for root, dirs, files in os.walk("ins"):   ### In place of ins, give the corresponding directory name where all the PDF's are stored
        #print("Came inside ins")
        for file in files:
            ##print(file)
            file=file.lower()
            if file.endswith(".pdf"):
                ##print("Came in")
                loader = PDFMinerLoader(os.path.join(root, file))
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                texts = text_splitter.split_documents(documents)
                #print("Here")
                ###print(f"Extracted {len(texts)} text chunks from {file}")
                all_texts.extend(texts)

    # create embeddings here
    ##print("Length of all_texts is",len(all_texts))
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(all_texts, embeddings, persist_directory=persist_directory)
    db.persist()

    end = time.time()
    print("Total time taken to ingest all files is ", end - start)
    print("Total files ingested =", len(files))

if __name__ == "__main__":
    main()
