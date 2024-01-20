# Step-by-Step Guide to Setting Up a Retrieval-Augmented Generation (RAG) System

In this article, we'll guide you through setting up your own Retrieval-Augmented Generation (RAG) system. This system allows you to upload your PDFs and ask a Language Model (LLM) about the information within those PDFs. The focus of this tutorial is on the blue section of the diagram, specifically not integrating Gradio at this stage. The tech stack involved includes:

- LLM: Llama2
- LLM API: llama.cpp service
- Langchain
- Vector DB: ChromaDB
- Embedding: sentence-Transformers

The cornerstone of this setup is Langchain, a framework for developing applications supported by language models. Langchain acts as a glue, offering various interfaces to connect LLM models with other tools and data sources. However, it's worth noting that Langchain is rapidly evolving, with frequent documentation and API updates. Below, we demonstrate the simplest way to set this up.

## Step 1: Environment Setup

Set up your python environment. In this tutorial, I used conda to create the environment and installed the following libraries in a Jupyter environment.

```python
 pip install -r requirements.txt  
```

## Step 2: File Processing and Database Import
We start by processing external information and storing it in the database for future knowledge queries. This step corresponds to the orange section of the diagram, specifically 1. Text Splitter and 2. Embedding.

### a) Using Document Loaders
Langchain offers around 55 types of document loaders, including loaders for Word, CSV, PDF, GoogleDrive, and YouTube. Here, we use PyMuPDFLoader to read in a resume. Note that PyMuPDF needs to be installed to use PyMuPDFLoader.

```python
from langchain.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("Virtual_characters.pdf")
PDF_data = loader.load()
```

### b) Using Text Splitter to Split Documents
Text splitter splits documents or text into chunks to avoid exceeding the LLM's token limit. For this, we use RecursiveCharacterTextSplitter or CharacterTextSplitter. The main parameters include chunk_size (determining the max number of characters per chunk) and chunk_overlap (specifying the overlapping characters between consecutive chunks).

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=5)
all_splits = text_splitter.split_documents(PDF_data)
```

### c) Loading the Embedding Model
We use Embedding to convert the chunks of text from step b) into vectors. LangChain provides interfaces for many Embedding models.

```python
from langchain.embeddings import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}
embedding = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
```

### d) Importing the Embedding Results into VectorDB
We store the results of the Embedding in VectorDB, using Chroma for implementation.

```python
# Embed and store the texts
persist_directory = 'db'
vectordb = Chroma.from_documents(documents=all_splits, embedding=embedding, persist_directory=persist_directory)
```

## Step 3: Initiating the LLM Service
There are two ways to initiate your LLM model and connect it to LangChain: using LangChain's LlamaCpp interface or setting up Llama2's API service through another method, such as llama.cpp's server. Here, both methods are demonstrated.

### a) Using LangChain's LlamaCpp
This method is simpler and involves using LlamaCpp's interface to load the model and initiate Llama's service.

```python
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
model_path = "llama.cpp/models/llama-2-7b-chat/llama-2_q4.gguf"

llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=100,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
)
```

### b) Using an Established API Service
If you have an established LLM API service, you'll need to use LangChain's ChatOpenAI interface.

```python
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(openai_api_key='None', openai_api_base='http://127.0.0.1:8080/v1')
```

## Step 4: Setting Your Prompt
Some LLMs can use specific prompts. Here, we use ConditionalPromptSelector to set prompts based on the model type.

```python
from langchain.chains import LLMChain
from langchain.chains.prompt_selector import ConditionalPromptSelector
from langchain.prompts import PromptTemplate

DEFAULT_LLAMA_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""<<SYS>> \n You are an assistant tasked with improving Google search \
results. \n <</SYS>> \n\n [INST] Generate THREE Google search queries that \
are similar to this question. The output should be a numbered list of questions \
and each should have a question mark at the end: \n\n {question} [/INST]""",
)

DEFAULT_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an assistant tasked with improving Google search \
results. Generate THREE Google search queries that are similar to \
this question. The output should be a numbered list of questions and each \
should have a question mark at the end: {question}""",
)

QUESTION_PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=DEFAULT_SEARCH_PROMPT,
    conditionals=[(lambda llm: isinstance(llm, LlamaCpp), DEFAULT_LLAMA_SEARCH_PROMPT)],
)

prompt = QUESTION_PROMPT_SELECTOR.get_prompt(llm)

llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "What is Taiwan known for?"
llm_chain.invoke({"question": question})
```

## Step 5: Text Retrieval + Query LLM
We've stored PDF information in the database and initiated the LLM service. Now, we connect the entire RAG process:

1. User sends a QA.
2. Text Retrieval from the database.
3. Combine QA with Text Retrieval and send to LLM.
4. LLM responds based on the information.
First, create a Retriever that returns corresponding documents based on unstructured QA. Then, combine Retriever, QA, and llm using RetrievalQA.

```python
retriever = vectordb.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)
```

## Step 6: Using Your RAG
We've connected the entire RAG process. Let's query about the virtual character Alison Hawk from the PDF records.

```python
query = "Tell me about Alison Hawk's career and age"
qa.invoke(query)
```
