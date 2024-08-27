from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader,TextLoader
import os 
from Embedding import Embedding
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv() # loading the environment variables 

class RAG:
    
    def __init__(self):
        #loading the documents.
        self.docs = self.load_documents(f'{os.getcwd()}/archive/Culture/')
        self.spillts = self.splitting_docs(self.docs)
        
        self.API_KEY=os.getenv('API_KEY')
        if self.API_KEY is None:
            self.API_KEY = 'AIzaSyC83XtGvVeMBM2RHvjNoeP89F5RBnLEs9w' 
        self.model = GoogleGenerativeAI(google_api_key=self.API_KEY,model='gemini-1.5-flash')
        embedding_model = self.embed_documents() ## convwerting documents into a vectors and store them into datastore
        self.retriever = embedding_model.as_retriever(search_type='similarity',search_kwargs={'k':2}) ## making the retriever
    
    def load_documents(self,dir):
        loader = DirectoryLoader(dir, loader_cls=TextLoader,loader_kwargs={'encoding':'utf-8'})
        return loader.load()
    
    def splitting_docs(self,docs,chunk_size=1500,chunk_overlap=150):
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap, add_start_index=True)
        return splitter.split_documents(docs) 
    
    def embed_documents(self):
        return Chroma(persist_directory='./vector_datastore').from_documents(embedding=Embedding(),documents=self.spillts)
        
    
    def format_docs(self,docs):
        return "\n".join(doc.page_content for doc in docs)
    
    
    def generate_text(self,question):
        template = '''
            انت مساعدي في الذكاء الاصطناعي سأعطيك بعض الأسئلة و بناء على المعلومات التي لديك قم بالاجابه 
            السؤال : {question}
            السياق : {context}
            الإجابه : 
        '''
        prompt = PromptTemplate(template=template,input_variables=['question','context'])
        rag_chain = (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.model
            | StrOutputParser()
        )
        return rag_chain.invoke(question)