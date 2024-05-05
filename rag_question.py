from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_qa_with_sources_chain
from langchain.chains import ConversationalRetrievalChain
from langchain_core.documents.base import Document
from langchain_core.prompts import ChatPromptTemplate

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.storage import LocalFileStore

from py_pdf_parser.loaders import load_file

from document_extraction.extract_pdf_data import parse_document_elements, event_entry_to_string

import json

def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def load_documents(fname):
    documents = load_file(fname)
    documents = parse_document_elements(documents)
    langchain_documents = []
    for document in documents:
        ldoc = Document(page_content=event_entry_to_string(document), metadata={
            "title": document["title"], 
            "organizer": document["organizer"], 
            "site": document["site"],
            "district": document["address"]["district"], 
            "street": document["address"]["street"]
            }
        )
        langchain_documents.append(ldoc)
    return langchain_documents

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

openai_key = load_json("openai_key.json")["key"]
documents = load_documents("original_program.pdf")

# Step 3
embeddings_model = OpenAIEmbeddings(api_key=openai_key)
embeddings_store = LocalFileStore("./embeddings_cache/")
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    embeddings_model, embeddings_store, namespace=embeddings_model.model
)
db = FAISS.from_documents(documents, cached_embeddings)

# Step 4
retriever = db.as_retriever()

# Step 5
llm_src = ChatOpenAI(temperature=0, model="gpt-4-turbo", openai_api_key=openai_key)

"""
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)"""

retrieval_qa = ConversationalRetrievalChain.from_llm(
    llm_src,
    retriever,
    return_source_documents=True,
)

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

top_recommendations = "seven"
#interests = "culture, technology, art, military, psychology and Italy"
interests = "flowers, fashion"


#prompt = f"Suggest {top_recommendations} unusual and unique events to visit in Warszawa (Warsaw) during Museums Night.\nSuggest ONLY events from Museums Night from the context.\nStop answering if the suggested entry is not in Museums Night event. Do not hallucinate facts. If there are no events relative to my interests, suggest events that are in the context, similar to my interests or something unusual, make sure to reach {top_recommendations} suggestions. For each entry return its title, the location, a summary, the reason why you suggested it."
#prompt = f"Suggest {top_recommendations} events to visit in Warszawa (Warsaw) during Museums Night given that my interests are: {interests}.\nSuggest ONLY events from Museums Night from the context.\nStop answering if the suggested entry is not in Museums Night event. Do not hallucinate facts. If there are no events relative to my interests, suggest events that are in the context, similar to my interests or something unusual, make sure to reach {top_recommendations} suggestions. For each entry return its title, the location, a summary, the reason why you suggested it."

contextualized_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", '''You are a helpful assistant that translates retrieves context information for Museums Night event in Warsaw (Warszawa) city of Poland and suggests events based on this information.
        You suggest ONLY events from Museums Night from the context.
        You do not hallucinate facts outside of the retrieved context, and stop answering if the suggested entry is not in the context.
        If there are no events relative to user's interests, you can suggest events that are in the context, similar to user's interests or something unusual.
        For each entry, you return its title, the location, a summary, the reason why you suggested it.
        
        CONTEXT:
        {context}'''),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

user_input = f"Suggest {top_recommendations} events to visit in Warszawa during Museums Night given that my interests are: {interests}. I want only events that do not require prior registration."

question_answer_chain = create_stuff_documents_chain(llm_src, contextualized_prompt)

rag_chain = create_retrieval_chain(retriever, question_answer_chain)

output = rag_chain.invoke({
    "input": user_input,
    "chat_history": []
})

print(f"Question:\n{user_input}\n\n")
print(f"Answer:\n{output['answer']}\n\n")
print("Sources:")
for source in output['context']:
    metadata = source.metadata
    print(f"{metadata['title']} - {metadata['organizer']} - {metadata['site']} - {metadata['district']} - {metadata['street']}")