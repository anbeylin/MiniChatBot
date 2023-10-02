from langchain.prompts import PromptTemplate

from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS
from langchain.memory import VectorStoreRetrieverMemory

from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.llms import OpenAI

from langchain.chains import LLMChain


import faiss
import json
import openai
import os


SETTINGS = "settings.json"


# ----------------------------------------------------------------------
def GetSettings(settings_file):
    try:
        with open(settings_file, 'r') as file:
            settings = json.load(file)

    except (FileNotFoundError, json.JSONDecodeError):
        print("Settings file not found")

    # It would be great to add defaults, then check all loaded settings 
    # and fall back to default if something is wrong

    return settings["llm_settings"], settings["storage_settings"]


# ----------------------------------------------------------------------
def GetPromptTemplate():
    CONV_TEMPLATE = """The following is a friendly conversation between a human and an AI. 
    If the AI does not know the answer to a question, it truthfully says it does not know.

    Pieces of previous conversation retrueved from history. 
    If these are not useful or relevan, then ignore history and reply to the new input.
    {history}

    Current conversation:
    Human: {input}
    AI:
    """

    return PromptTemplate(input_variables=["history", "input"], template=CONV_TEMPLATE)


# ----------------------------------------------------------------------
def InitModel(llm_settings):
    # Need to set API KEY in environment for things to work properly
    openai.api_key = llm_settings["open_ai_api_key"]
    os.environ["OPENAI_API_KEY"] = openai.api_key

    # Initialize LLM
    # In the future we can pass prompt template to this function as a parameter
    llm = OpenAI(
            openai_api_key=llm_settings["open_ai_api_key"], 
            model_name=llm_settings["model"], 
            temperature=llm_settings["temperature"])

    # Very somple chain
    chat_bot_chain = LLMChain(prompt=GetPromptTemplate(), llm=llm)

    return chat_bot_chain

# ----------------------------------------------------------------------
def InitStorage(storage_settings):
    embedding_size = 1536 # Dimensions of the OpenAIEmbeddings
    index = faiss.IndexFlatL2(embedding_size) # similarity function is L2 distance
    embeddings = OpenAIEmbeddings()

    try: # try loading storage
        vectorstore = FAISS.load_local(storage_settings["local_storage_name"], embeddings)
    except RuntimeError: # If problems create a new vector storage
        vectorstore = FAISS(embeddings.embed_query, index, InMemoryDocstore({}), {})

    # Parameter k gives how many messages to retrieve from history 
    retriever = vectorstore.as_retriever(search_kwargs=dict(k=storage_settings["retrieval_msg_count"]))

    memory = VectorStoreRetrieverMemory(retriever=retriever)

    return memory, vectorstore


# ----------------------------------------------------------------------
def AskChatBot(chat_bot_chain, memory, user_input):
    retrieved_messages = memory.load_memory_variables({"prompt": user_input})

    # this template_input is used by the prompt template
    # Important! Keys (history, input) are the same as in prompt template
    template_input = {"history":retrieved_messages["history"], "input":user_input}

    # This is where magic happens
    chat_bot_answer = chat_bot_chain.run(template_input)

    return chat_bot_answer


# ----------------------------------------------------------------------
def SaveConversation(memory, user_input, chat_bot_answer):
    memory.save_context({"input": user_input}, {"output": chat_bot_answer})


# ----------------------------------------------------------------------
def PersistMemoryLocal(vectorstore, local_memory_name):
    vectorstore.save_local(local_memory_name)