# LangChain Fundamentals

## Introduction
LangChain is an open-source framework designed for building applications that leverage Large Language Models (LLMs). It provides essential tools for integrating external data sources, memory management, and chaining different LLM calls to create powerful AI applications.

## Features
- **LLM Wrappers**: Easy integration with OpenAI, Hugging Face, and other LLM providers.
- **Prompt Templates**: Standardize and optimize prompts.
- **Chains**: Sequence multiple calls to LLMs and tools.
- **Memory**: Store and retrieve conversational context.
- **Agents & Tools**: Use LLMs as reasoning agents connected to external tools (e.g., APIs, databases).
- **Retrieval-Augmented Generation (RAG)**: Enhance responses with real-time external knowledge.

## Installation
```bash
pip install langchain openai
```
For additional dependencies:
```bash
pip install langchain[all]
```

## Basic Usage
### 1. Initialize an LLM
```python
from langchain.llms import OpenAI
llm = OpenAI(model_name="gpt-3.5-turbo", api_key="your-api-key")
response = llm("Explain LangChain in one sentence.")
print(response)
```

### 2. Using Prompt Templates
```python
from langchain.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms."
)
print(template.format(topic="Machine Learning"))
```

### 3. Creating a Chain
```python
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=template)
response = chain.run("Artificial Intelligence")
print(response)
```

### 4. Memory in Conversational Agents
```python
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory()
memory.save_context({"input": "Hello!"}, {"output": "Hi, how can I help?"})
print(memory.load_memory_variables({}))
```

## Advanced Usage
- **Integrating with APIs** (Google Search, Wolfram Alpha, etc.)
- **Connecting to Vector Databases** (Pinecone, Chroma, FAISS)
- **Deploying LangChain Apps** (FastAPI, Streamlit, Flask)
