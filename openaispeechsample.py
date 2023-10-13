import json
import os
import azure.cognitiveservices.speech as speechsdk
import openai
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv()

# This example requires environment variables named "OPEN_AI_KEY" and "OPEN_AI_ENDPOINT"
# Your endpoint should look like the following https://YOUR_OPEN_AI_RESOURCE_NAME.openai.azure.com/
openai.api_key = os.environ.get('AZURE_OPEN_AI_KEY')
openai.api_base =  os.environ.get('AZURE_OPEN_AI_ENDPOINT')
openai.api_type = 'azure'
openai.api_version = '2023-05-15'

# This will correspond to the custom name you chose for your deployment when you deployed a model.
deployment_id='druanaoaitextdavinci003' 

# This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'), region=os.environ.get('SPEECH_REGION'))
audio_output_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)

# Should be the locale for the speaker's language.
speech_config.speech_recognition_language="en-US"
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

# The language of the voice that responds on behalf of Azure OpenAI.
speech_config.speech_synthesis_voice_name='en-US-JennyMultilingualNeural'
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output_config)

# Use the current user identity to authenticate with Azure OpenAI, Cognitive Search and Blob Storage (no secrets needed, 
# just use 'az login' locally, and managed identity when deployed on Azure). If you need to use keys, use separate AzureKeyCredential instances with the 
# keys for each service
AZURE_SEARCH_KEY = os.environ.get('AZURE_SEARCH_KEY')
azure_search_credential = AzureKeyCredential(AZURE_SEARCH_KEY)

# Set up clients for Cognitive Search and Storage
AZURE_SEARCH_SERVICE = os.environ.get('AZURE_SEARCH_SERVICE_NAME')
print(AZURE_SEARCH_SERVICE)
AZURE_SEARCH_INDEX = os.environ.get('AZURE_SEARCH_INDEX_NAME')
search_client = SearchClient(
    endpoint=f"https://{AZURE_SEARCH_SERVICE}.search.windows.net",
    index_name=AZURE_SEARCH_INDEX,
    credential=azure_search_credential)


# Chat roles
SYSTEM = "system"
USER = "user"
ASSISTANT = "assistant"

system_message_chat_conversation = """Assistant helps the company employees with their healthcare plan questions, and questions about the employee handbook. Be brief in your answers.
Answer ONLY with the facts listed in the list of sources below. If there isn't enough information below, say you don't know. Do not generate answers that don't use the sources below. If asking a clarifying question to the user would help, ask the question.
Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. Use square brackets to reference the source, e.g. [info1.txt]. Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf].
"""
chat_conversations = [{"role" : SYSTEM, "content" : system_message_chat_conversation}]

summary_prompt_template = """
Given the chat history and user question generate a search query that will return the best answer from the knowledge base.
Try and generate a grammatical sentence for the search query.
Do NOT use quotes and avoid other search operators.
Do not include cited source filenames and document names such as info.txt or doc.pdf in the search query terms.
Do not include any text inside [] or <<>> in the search query terms.
If the question is not in English, translate the question to English before generating the search query.

Search query:
"""
query_summary_conversations = [{"role" : SYSTEM, "content" : summary_prompt_template}]

AZURE_OPENAI_CHATGPT_DEPLOYMENT=os.environ.get('AZURE_OPENAI_CHATGPT_DEPLOYMENT')
AZURE_OPENAI_CHATGPT_MODEL=os.environ.get('AZURE_OPENAI_CHATGPT_MODEL')
KB_FIELDS_SOURCEPAGE=os.environ.get('KB_FIELDS_SOURCEPAGE')
KB_FIELDS_CONTENT=os.environ.get('KB_FIELDS_CONTENT')
AZURE_OPENAI_EMB_DEPLOYMENT=os.environ.get('AZURE_OPENAI_EMB_DEPLOYMENT')
# Prompts Azure OpenAI with a request and synthesizes the response.
def ask_openai(prompt):

    query_summary_conversations.append({"role": USER, "content": prompt })
    exclude_category = None
    query_completion = openai.ChatCompletion.create(
        deployment_id=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
        model=AZURE_OPENAI_CHATGPT_MODEL,
        messages=query_summary_conversations, 
        temperature=0.7, 
        max_tokens=1024, 
        n=1)
    search = query_completion.choices[0].message.content

    query_vector = openai.Embedding.create(engine=AZURE_OPENAI_EMB_DEPLOYMENT, input=search)["data"][0]["embedding"]
    # Search Azure Cognitive Search
    print("Searching:", search)
    print("-------------------")
    filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None
    r = search_client.search(search, 
                         filter=filter,
                         query_type=QueryType.SEMANTIC, 
                         query_language="en-us", 
                         query_speller="lexicon", 
                         semantic_configuration_name="default", 
                         top=2,
                         vector=query_vector if query_vector else None, 
                         top_k=5 if query_vector else None, 
                         vector_fields="embedding" if query_vector else None)
    results = [doc[KB_FIELDS_SOURCEPAGE] + ": " + doc[KB_FIELDS_CONTENT].replace("\n", "").replace("\r", "") for doc in r]
    print("Search results:" + results[0])
    print("\n")
    content = "\n".join(results)
    user_content = prompt + " \nSOURCES:\n" + content
    chat_conversations.append({"role": USER, "content": user_content })

    # Ask Azure OpenAI

    response = openai.ChatCompletion.create(
        deployment_id=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
        model=AZURE_OPENAI_CHATGPT_MODEL,
        messages=chat_conversations, 
        temperature=0.7, 
        max_tokens=1024, 
        n=1)
    
    text = response.choices[0].message.content
    """
    response = openai.Completion.create(engine=deployment_id, prompt=prompt, max_tokens=100)
    text = response['choices'][0]['text'].replace('\n', ' ').replace(' .', '.').strip()
    """
    print('Azure OpenAI response:' + text)

    # Azure text to speech output
    speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()

    # Check result
    if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized to speaker for text [{}]".format(text))
    elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_synthesis_result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))

# Continuously listens for speech input to recognize and send as text to Azure OpenAI
def chat_with_open_ai():
    while True:
        print("Azure OpenAI is listening. Say 'Stop' or press Ctrl-Z to end the conversation.")
        try:
            # Get audio from the microphone and then send it to the TTS service.
            speech_recognition_result = speech_recognizer.recognize_once_async().get()

            # If speech is recognized, send it to Azure OpenAI and listen for the response.
            if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
                if speech_recognition_result.text == "Stop.": 
                    print("Conversation ended.")
                    break
                print("Recognized speech: {}".format(speech_recognition_result.text))
                ask_openai(speech_recognition_result.text)
            elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
                print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
                break
            elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = speech_recognition_result.cancellation_details
                print("Speech Recognition canceled: {}".format(cancellation_details.reason))
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    print("Error details: {}".format(cancellation_details.error_details))
        except EOFError:
            break

# Main

try:
    chat_with_open_ai()
except Exception as err:
    print("Encountered exception. {}".format(err))