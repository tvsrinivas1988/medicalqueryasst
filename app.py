import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
import pinecone
from langchain_nomic import NomicEmbeddings
from langchain import hub
import pprint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pinecone import Pinecone
from langchain_core.runnables import RunnableLambda


load_dotenv()
def format_docs(docs):
    return "\n\n".join(doc for doc in docs)

format_docs_runnable = RunnableLambda(format_docs)


GROQ_API_KEY=st.secrets["GROQ_API_KEY"]
LANGSMITH_API_KEY=st.secrets["LANGSMITH_API_KEY"]
LANGSMITH_PROJECT=st.secrets["LANGSMITH_PROJECT"]
LANGSMITH_TRACING_V2=st.secrets["LANGSMITH_TRACING_V2"]
LANGSMITH_PROJECT=st.secrets["LANGSMITH_PROJECT"]
PINECONE_API_KEY=st.secrets["PINECONE_API_KEY"]
NOMIC_API_KEY=st.secrets["NOMIC_API_KEY"]

embeddings=NomicEmbeddings(model="nomic-embed-text-v1.5",dimensionality=350)
pc = Pinecone( api_key=st.secrets["PINECONE_API_KEY"] ) 
index = pc.Index("medicalqueryasst")
st.set_page_config(page_title="Medical Query RAG Assistant")
st.title("Medical Query RAG Assistant")
st.write("Ask questions about drug labels, efficacy, etc.")
query = st.text_input("Enter your question:")
if query:
    with st.spinner("Retrieving info and generating answer..."):
        llm = ChatGroq(
        model="deepseek-r1-distill-llama-70b",
        temperature=0,
        max_tokens=None,
        reasoning_format="parsed",
        timeout=None,
        max_retries=2);
        query_vector = embeddings.embed_query(query)
        print(len(query_vector))

        res = index.query(vector=query_vector, top_k=3, include_metadata=True)
       ##print(res.matches)
        contexts = [match['metadata'].get("text", "") for match in res['matches']]
        context_str = "\n\n".join(contexts)
        print(contexts)
        prompt = hub.pull("rlm/rag-prompt")
        rag_chain = (
            {"context": format_docs_runnable , "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        ans = rag_chain.invoke({"context": contexts, "question": query})
    st.subheader("Answer")
    st.write(ans)




