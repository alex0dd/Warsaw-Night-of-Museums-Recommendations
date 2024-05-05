import argparse
import json
import os

from langchain.chains import RetrievalQA
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

from utils import load_documents, load_json, write_suggestions_file


def parse_args():
    parser = argparse.ArgumentParser()

    # Integer argument for the number of top recommendations
    parser.add_argument(
        "--top_recommendations",
        type=int,
        default=5,
        help="Number of top recommendations to return (default: 5)",
    )

    # String argument for interests; expects a comma-separated list
    parser.add_argument(
        "--interests",
        type=str,
        default="fashion, colors",
        help='List of interests separated by commas (default: "fashion, colors")',
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output file where results will be written",
    )

    args = parser.parse_args()
    return args


args = parse_args()

output_path = args.output
top_recommendations = args.top_recommendations
interests = args.interests
query = f"Suggest {top_recommendations} events that do not require prior registration, to visit during Museums Night given that my interests are: {interests}."

# Envs
llm_model_name = os.environ.get("LLM_MODEL", "gpt-4-turbo")
openai_key_path = os.environ.get("OPENAI_KEY_PATH", "keys/openai_key.json")

openai_key = load_json(openai_key_path)["key"]
documents = load_documents("data/original_program.pdf")

# Instantite embeddings and retrieval
embeddings_model = OpenAIEmbeddings(api_key=openai_key)
embeddings_store = LocalFileStore("data/embeddings_cache/")
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    embeddings_model, embeddings_store, namespace=embeddings_model.model
)
db = FAISS.from_documents(documents, cached_embeddings)
retriever = db.as_retriever()

# Instantiate LLM
llm_src = ChatOpenAI(temperature=0, model=llm_model_name, openai_api_key=openai_key)

# Instantiate retrieval chain
contextualized_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """You are a helpful assistant that retrieves context information for Museums Night event in Warsaw (Warszawa) city of Poland and suggests events based on this information.
        You suggest ONLY events from Museums Night from the context.
        You do not hallucinate facts outside of the retrieved context.
        You stop the current suggestion if the suggested entry is not in the context.
        The user specifies the amount of events they want to attend and you suggest them exactly this amount of events.
        If there are no events relative to user's interests, you suggest events that are in the context, either similar to user's indicated interests or unusual for the user.
        For each entry, you return its title, the location, the opening hours, a summary, the reason why you suggested it.
        
        Question: {question}
        Context: {context}
        Answer:""",
        ),
    ]
)
retrieval_qa = RetrievalQA.from_chain_type(
    llm_src,
    retriever=retriever,
    chain_type_kwargs={"prompt": contextualized_prompt},
    return_source_documents=True,
)

# Run the pipeline to get the answer
output = retrieval_qa({"query": query})

write_suggestions_file(output, output_path)
