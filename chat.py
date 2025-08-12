from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from rag_chat import carregar_arquivos, split_data, criar_vectordb


documentos = carregar_arquivos()
documentos = split_data(documentos)
vectordb = criar_vectordb(documentos)

# Modelo de chat
chat = ChatOpenAI(model="gpt-4o-mini")

# Prompt do chatbot
chain_prompt = PromptTemplate.from_template(
    """Utilize o contexto fornecido para responder à pergunta no final.
    Se não souber, diga que não sabe.  

    Contexto: {context}

    Pergunta: {question}

    Resposta:
    """
)

# Cadeia de QA com recuperação
chat_chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=vectordb.as_retriever(search_type="mmr"),
    chain_type_kwargs={"prompt": chain_prompt},
    return_source_documents=True
)

# --- Loop de interação ---
print("Digite sua pergunta (ou 'exit' para sair)")

while True:
    pergunta = input("\nPergunta: ").strip()
    if pergunta.lower() in ['exit', 'out']:
        print("Saindo...")
        break
    
    resposta = chat_chain.invoke({'query': pergunta})
    print("Resposta:", resposta['result'])
