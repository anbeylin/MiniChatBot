#!/usr/bin/env python3

# all functions are defined in the chatbothelper
from chatbothelper import *


# ----------------------------------------------------------------------
# Main logic here is to very simply encasulate steps of the process in the separate functions.
# In the future this will let us supply different versions of LLM, memory, storage.

def main():
    print("You are talking with a simple chatbot with memory. Please enter a message or type 'exit' to stop.\n")

    # load settings from json file SETTINGS
    llm_settings, storage_settings = GetSettings(SETTINGS)
    
    # Initialize OpenAI API, model and chain
    chat_bot_chain = InitModel(llm_settings)
    
    # Load memory from local storage or create new one
    # Note that memory object stores input and answers in-memory during conversation
    # at the end vectorstore will be used to persist it on disk
    memory, vectorstore = InitStorage(storage_settings)

    ### --- Main conversation loop ---
    while True:
        new_user_input = input("Please enter a message or type 'exit' to stop: \n")

        if new_user_input.lower() == 'exit':
            print("Exiting program...")
            break

        chat_bot_answer = AskChatBot(chat_bot_chain, memory, new_user_input)

        print("Simple Chat Bot:\n", chat_bot_answer)

        # Keep current conversation in memory
        SaveConversation(memory, new_user_input, chat_bot_answer)
    ### --- End of conversation loop ---

    # persist conversation on disk
    PersistMemoryLocal(vectorstore, storage_settings["local_storage_name"])


if __name__ == "__main__":
    main()