import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.chat_models import ChatOllama
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

#  Access the "nomic-embed-text" embedding model
embedding_model = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
# Access the "mistral" model
llm = ChatOllama(model="mistral")
prompt = hub.pull("rlm/rag-prompt")
# Indexing

def load(doc_url):
    # Load, chunk and index the contents of the blog.
    loader = WebBaseLoader(
        web_paths=(doc_url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    return docs

def split(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return splits

def store(splits):
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model,
    collection_name="local-ai")
    return vectorstore

# Retrieval & Generation

def retrieve(vectorstore):
    return vectorstore.as_retriever()

def generate(retriever, query):

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    return rag_chain.invoke(query)

# Formatting Document
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == '__main__':
    docs = load("https://lilianweng.github.io/posts/2023-06-23-agent/")
    splits = split(docs)
    vectorstore = store(splits)
    retriever = retrieve(vectorstore)
    response= generate(retriever,"What is the document all about?")
    print(response)

