import vecs
from vecs.adapter import Adapter, TextEmbedding
import os
from dotenv import load_dotenv
import google.generativeai as genai
import os
import re
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()
DB_CONNECTION = os.getenv("DB_CONNECTION")
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# create vector store client
vx = vecs.create_client(DB_CONNECTION)

def remove_generic_words(text):
    generic_words = r"\b(the|is|was|were|when|so|therefore|since|what|where|of|be|to|are|which|why)\b"
    return re.sub(generic_words, "", text, flags=re.IGNORECASE).strip()
    
def generate_response(question,vector_db):
    good_question=remove_generic_words(question)
    print(good_question+"\n")
    response=vector_db.query(
        data=good_question,          # required
        limit=5,                     # number of records to return
        filters={},                  # metadata filters
        measure="cosine_distance",   # distance measure to use
        include_value=False,         # should distance measure values be returned?
        include_metadata=True,      # should record metadata be returned?
    )
    # Loop through the list and extract "text"
    output=""
    for index, details in response:
        output+=details["text"]
    answer = model.generate_content(f"You are a customer support assistance from the organisation of Zomato, a food delivery service. Use first person like 'We' to answer queries. Don't answer beyond the given context. The context is :{output}. Customer Question: {question}")
    return {"question":question,"retrieved_data":answer.candidates[0].content.parts[0].text}

# Initialize FastAPI app
app = FastAPI()
# Pydantic model for request
class QueryRequest(BaseModel):
    question: str
    database_name: str  # User specifies which vector DB to query

@app.post("/query")
async def query_vector_db(request: QueryRequest):
    # Get the vector database collection specified by the user
    vecs_db= vx.get_or_create_collection(
    name=request.database_name,
    adapter=Adapter(
        [
        #ParagraphChunker(skip_during_query=True),
        TextEmbedding(model='all-MiniLM-L6-v2', batch_size=8)#384 dimensions 
        ]
    ),
    dimension=384
    )
    # Generate response
    response = generate_response(request.question, vecs_db)

    return response

