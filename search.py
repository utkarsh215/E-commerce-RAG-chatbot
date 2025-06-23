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

# Serialize function for product documents (used before storing/embedding, not shown storing here)
def serialize_product_for_embedding(doc):
    title = doc.get("title", "No Title")
    short_desc = doc.get("shortDescription", "")
    detailed_desc = doc.get("detailedDescription", "")
    category = ", ".join(doc.get("category", []))
    features = ", ".join(doc.get("features", []))
    specs = doc.get("specifications", {})
    weight = specs.get("weight", "N/A")
    color = specs.get("color", "N/A")
    warranty = specs.get("warranty", "N/A")
    pros_list = doc.get("pros", [])
    pros = "\n  - " + "\n  - ".join(pros_list) if pros_list else "None"
    cons_list = doc.get("cons", [])
    cons = "\n  - " + "\n  - ".join(cons_list) if cons_list else "None"
    usage = ", ".join(doc.get("usageScenarios", []))
    price = doc.get("price", "N/A")
    brand = doc.get("brand", "Unknown Brand")
    rating = doc.get("rating", "No Rating")
    review_summary = doc.get("reviewSummary", "No review summary available.")
    keywords = ", ".join(doc.get("keywords", []))

    reviews = doc.get("reviews", [])
    sorted_reviews = sorted(
        reviews,
        key=lambda r: datetime.fromisoformat(r.get("createdAt", "1970-01-01T00:00:00")),
        reverse=True
    )

    recent_review_per_rating = {}
    for r in sorted_reviews:
        r_rating = r.get("rating")
        if r_rating in [1, 2, 3, 4, 5] and r_rating not in recent_review_per_rating:
            recent_review_per_rating[r_rating] = r

    review_text_sections = []
    for rating_level in range(5, 0, -1):
        if rating_level in recent_review_per_rating:
            rev = recent_review_per_rating[rating_level]
            reviewer = rev.get("user", {}).get("name", "Anonymous")
            title_r = rev.get("title", "")
            body = rev.get("body", "")
            review_text_sections.append(
                f"⭐ {rating_level}-Star Review by {reviewer}:\nTitle: {title_r}\nBody: {body}"
            )

    review_text = "\n\n".join(review_text_sections) if review_text_sections else "No user reviews available."

    created_at = doc.get("createdAt", "")
    updated_at = doc.get("updatedAt", "")

    return f"""Product: {title}
Short Description: {short_desc}
Detailed Description: {detailed_desc}
Category: {category}
Features: {features}

Specifications:
  - Weight: {weight}
  - Color: {color}
  - Warranty: {warranty}

Pros:{pros}

Cons:{cons}

Usage Scenarios: {usage}
Price: ${price}
Brand: {brand}
Rating: {rating} stars
Review Summary: {review_summary}

Top Recent Reviews by Rating:
{review_text}

Keywords: {keywords}
Created At: {created_at}
Last Updated: {updated_at}
"""
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

