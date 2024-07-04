import streanlit as st 
from streamlit_chat import message
from lanmgchain.chain import ConversationalRetrivalChain 
from langchain.document_loaders import pyPDFLoader, Directory, DirectoryLoader
from langchain.embedding import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from lanchain.vectorstores import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory  

#load the pdf
loader = DirectoryLoader("data/",globe="*.pdf",loader_cls=pyPDFLoader)
documents = loader.load()

#split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunks_size=500,chunk_overlap=50)

#create embeddings
embeddings = HuggingFaceEmbeddings(model_name="")
    
#vectorstore
vector_store = FAISS.from_documents(text_chunks,embeddings)

#create llm
llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin",model_type="llama",
                    config={'max_new_tokens':128,'temperature':0.01})    

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chain = ConversationalRetrievalChain.from_llm(llm=llm,chain_type='stuff',
                                              retriever=vector_store.as_retriever(search_kwargs={"k":2}),
                                              memory=memory)
