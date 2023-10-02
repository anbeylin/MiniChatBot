## Mini Chat Bot
Mini Chat Bot is a simple conversational chatbot with limited memory capabilities, leveraging the power of OpenAI. It maintains a single conversation stream and does not differentiate memories into independent conversations. The project is designed to be run from the command line and uses FAISS as a vector database to persist memory locally on your hard drive.


### Features
- Single conversation stream memory.
- Uses OpenAI for conversation.
- Vector database for memory with FAISS.
- Persistence of memory on local storage.
- Easy configuration through settings.json.


### Getting Started
1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Install the required packages. *pip install -r requirements.txt*
4. Update the settings.json file with your OpenAI API key.
5. Run the project *python chatbot.py*


### Configuration (settings.json)
#### LLM Settings
- open_ai_api_key: Your OpenAI API key.
- model: Model name used by OpenAI. (Default: text-davinci-003)
- temperature: Controls the randomness of the model's output. (Default: 0.7)

#### Storage Settings
- local_storage_name: Name of the local directory where the chat memories will be saved. (Default: local_chat_memory)
- retrieval_msg_count: Number of messages to be retrieved from the memory. (Default: 4)


### Usage
Once the project is running, enter the messages when prompted, or type 'exit' to stop the program.