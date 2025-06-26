import os
from dotenv import load_dotenv
from datetime import datetime
import pymongo
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
DB_CONNECT = os.getenv("CONNECTION_DB")
GEMINI_API = os.getenv("GEMINI_API")

os.environ["GOOGLE_API_KEY"] = GEMINI_API

# Connect to MongoDB
client = pymongo.MongoClient(DB_CONNECT)
db = client.E_Commerce
collection = db.Products

# Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# LLM setup
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0, max_tokens=790)

# Vector search setup
vectorStore = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name="vector_index",
    text_key="embedding_text",
    embedding_key="embedding"
)
retriever = vectorStore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Custom prompt
system_prompt = (
    "You are a helpful assistant that provides product recommendations and information. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentences maximum.\n\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Memory to store conversation context
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Conversational RAG Chain
conversational_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=False
)

# Chat loop
print("Bot is ready! Type 'exit' to end the chat.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Bot: Goodbye!!")
        break

    result = conversational_chain.invoke({"question": user_input})
    # Safely access answer
    if isinstance(result, dict) and "answer" in result:
        print("Bot:", result["answer"])
    elif hasattr(result, "answer"):
        print("Bot:", result.answer)
    else:
        print("Bot:", result)

