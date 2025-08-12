import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Carrega variáveis de ambiente
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# --- Funções utilitárias ---

def carregar_arquivos():
    """Carrega arquivos de texto e retorna como lista de documentos."""
    caminhos = ["files/pg1228.txt"]
    documentos = []
    for caminho in caminhos:
        loader = TextLoader(caminho, encoding='utf-8')
        documentos.extend(loader.load())
    return documentos

def split_data(documentos):
    """Faz split de documentos em pedaços menores para melhor indexação."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    documentos = splitter.split_documents(documentos)
    for i, doc in enumerate(documentos):
        doc.metadata['source'] = doc.metadata.get('source', '').replace('arquivos/', '')
        doc.metadata['doc_id'] = i
    return documentos

def criar_vectordb(documentos):
    """Cria o carrega base vectorial"""

    diretorio_db = 'arquivos/chat_retrieval_db'
    embeddings_model = OpenAIEmbeddings()
    
    if os.path.exists(diretorio_db):
        return Chroma(persist_directory=diretorio_db, embedding_function=embeddings_model)
    
    vectordb = Chroma.from_documents(
        documents=documentos,
        embedding=embeddings_model,
        persist_directory=diretorio_db
    )
    return vectordb