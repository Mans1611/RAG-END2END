from sentence_transformers import SentenceTransformer

class Embedding:
  def __init__(self):
    self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
  def embed_documents(self,text):
    return self.model.encode(text).tolist()
  def embed_query(self,query):
    return self.model.encode(query).tolist()