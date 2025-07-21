import os
import time
import uuid
from dotenv import load_dotenv, find_dotenv                           
from types import SimpleNamespace

def load_env():
    _ = load_dotenv(find_dotenv())

def get_ai21_api_key():
    load_env()
    ai21_api_key = os.getenv("AI21_API_KEY")
    return ai21_api_key


def file_upload(client):
  text = open('Nvidia_10K_20240128.txt', 'r', encoding='utf-8').read()
  new_file_name = 'Nvidia_10K_2024_' + str(uuid.uuid4().hex) + '.txt'

  with open(new_file_name, 'w') as file:
      file.write(text)

  time.sleep(5)

  file_path = './' + new_file_name
  label = '10k_'+ str(uuid.uuid4().hex)

  file_id = client.library.files.create(
      file_path=file_path,
      labels=[label]
  )
  time.sleep(30)
  return file_id

def call_convrag(client, message):
    # Convert chat history to convrag messages format
    DEFAULT_RESPONSE = "I'm sorry, I cannot answer your questions based on the documents I have access to."

    try:
        chat_response = client.beta.conversational_rag.create(
            messages=message,
            query_extraction_model = 'jamba-1.5-large',
            question_answering_model = 'jamba-1.5-large',
            # labels=["10q"],
            # max_segments = 15,
            # retrieval_similarity_threshold = 0.8, # Range: 0.5 – 1.5
            # retrieval_strategy = 'segments',  # ['segments', 'add_neighbors', 'full_doc']
            # max_neighbors = 2, # Used only when retrieval_strategy = 'add_neighbors'
            # hybrid_search_alpha = 0.98 # Range: 0.0 – 1.0. 1.0 means using only dense embeddings; 0.0 means using only keyword search.
        )

    except Exception as e:
        raise Exception(f"Error occurred: {e}")

    if chat_response.context_retrieved and not chat_response.answer_in_context:
      # context_retrieved: [boolean] True if the RAG engine was able to find segments related to the user's query.
      # answer_in_context: [boolean] True if an answer was found in the provided documents.
        response = SimpleNamespace(choices=[SimpleNamespace(content=DEFAULT_RESPONSE)], sources=[SimpleNamespace(text="", file_name="")])
    else:
        response = chat_response

    return response
