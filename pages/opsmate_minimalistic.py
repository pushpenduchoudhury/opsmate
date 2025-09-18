import io
import os
import uuid
import socket
import chromadb
import traceback
import streamlit as st
from utils import utils
from pathlib import Path
import config.conf as conf
from dotenv import load_dotenv
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain_community.document_loaders import (
        TextLoader, 
        PyPDFLoader, 
        WebBaseLoader,
        Docx2txtLoader, 
        UnstructuredExcelLoader, 
        UnstructuredPowerPointLoader
    )
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"

# Set Title
logo_col, title_col, clear_col = st.columns([0.9, 11, 0.5])
with logo_col:
    st.image(image = str(Path(conf.ASSETS_DIR, "opsmate.gif")), width = "stretch")
with title_col:
    st.header("NextGen OpsMate", divider = "rainbow", anchor = False)
with clear_col:
    def clear_messages():
        st.session_state.voice_chat_messages = st.session_state.voice_chat_messages[:1]
    st.write("")
    st.write("")
    st.button(":material/mop:", on_click = lambda: clear_messages(), type = "secondary", width = "stretch", help = "Clear chat messages")

    
# Initialize chat session in streamlit
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

st.set_page_config(
        page_title = "NextGen OpsMate",
        page_icon = "🕵🏻",
        layout = "wide",
        initial_sidebar_state = "expanded"
    )


with st.sidebar:
    st.divider()
    
    st.subheader("⚙️ Settings")
    model_settings = st.expander("Model", icon = "🇦🇮")
    
    with model_settings:
        st.subheader(":grey[Provider]")
        model_provider = st.radio(":grey[Provider]", options = [conf.OLLAMA_MODEL_PROVIDER, conf.GOOGLE_MODEL_PROVIDER, conf.GROQ_MODEL_PROVIDER, conf.OPENAI_MODEL_PROVIDER], key = "model_provider", horizontal = True, index = 1, label_visibility = "collapsed")
    

def get_ollama_models() -> list:
    import ollama
    ollama_client = ollama.Client()
    llm = ollama_client.list()
    model_list:list[str] = [i['model'] for i in llm['models']]
    return model_list


if model_provider == conf.OLLAMA_MODEL_PROVIDER:
    try:
        ollama_models = get_ollama_models()
    except Exception as e:
        st.error(f"ERROR: {e}", icon = "⚠️")
        st.stop()

    if len(ollama_models) == 0:
        st.info(f"""No models found...! Please download a model from Ollama library to proceed.

Command:

ollama pull <model name>
                
You can visit the website: https://ollama.com/library to get models names.
""", icon = "⚠")
        st.stop()
    
    else:
        llm_models = ollama_models


elif model_provider == conf.GOOGLE_MODEL_PROVIDER:
    gemini_models = conf.GEMINI_MODELS
    
    if len(gemini_models) == 0:
        st.info("No Gemini models found...! Please configure a Gemini model to proceed.", icon = "⚠")
        st.stop()
    else:
        llm_models = gemini_models
    
elif model_provider == conf.GROQ_MODEL_PROVIDER:
    groq_models = conf.GROQ_MODELS
    
    if len(groq_models) == 0:
        st.info("No Groq models found...! Please configure a Gemini model to proceed.", icon = "⚠")
        st.stop()
    else:
        llm_models = groq_models

elif model_provider == conf.OPENAI_MODEL_PROVIDER:
    openai_models = conf.OPENAI_MODELS
    
    if len(openai_models) == 0:
        st.info("No OpenAI models found...! Please configure a Gemini model to proceed.", icon = "⚠")
        st.stop()
    else:
        llm_models = openai_models
    
else:
    st.error("Invalid model type selected. Please choose either Ollama or OpenAI.", icon = "🚫")
    st.stop()


def get_existing_vectordb(collection_name) -> Chroma:
    client = chromadb.Client(Settings(is_persistent = True, persist_directory = str(conf.CHROMADB_DIR)))
    vectordb = Chroma(client = client, collection_name = collection_name, embedding_function = embedding_function)
    return vectordb

def list_collections():
    client = chromadb.Client(Settings(is_persistent = True, persist_directory = str(conf.CHROMADB_DIR)))
    collections = [collection.name for collection in client.list_collections()]
    return collections


def get_history_aware_retriever(vector_db: Chroma, llm):
    """Create a history-aware retriever chain using the vector database and LLM."""
    # The prompt used to generate the search query for the retriever.
    retriever_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name = "chat_history"),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a search query to look up in order to get inforamtion relevant to the conversation, focusing on the most recent messages."),
            ])
    retriever = vector_db.as_retriever()
    retriever_chain = create_history_aware_retriever(llm, retriever, retriever_prompt)
    return retriever_chain

def get_retrieval_chain(vector_db: Chroma, llm):
    # Prompt template MUST contain input variable “context” (override by setting document_variable), which will be used for passing in the formatted documents.
    document_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are an IT Support assistant. You will have to answer to user's questions and issues which is relevant to the context provided.
						    Provide step by step guidance on the resolution of the user issues or questions. Don't give all the steps at one go if any relevant information is not found in the given context,
                            Try to gather more information from the user in such scenarios to see if it matches with any of the issues in the context. Even after asking for 2 or 3 clarifications if the context is not found then say no such issues were reported earlier and reply using your own knowledge.
						    You will have some context to help with your answers, but it might not always be completely related or helpful. 
						    **Never quote references from the document provided, respond in your own language.**
                            You can also use your knowledge to assist answering the user's propmts.
						    **Never answer question which is not related to the context provided.**
                            If any irrelevant questions are asked unrelated to the IT incidents, reply that you can only assist with IT incident issues.
                            {"**Respond only in one or two sentence**" if enable_voice else ""}
            {{context}}"""),
            MessagesPlaceholder(variable_name = "chat_history"),
            ("user", "{input}"),
        ])
    retriever = get_history_aware_retriever(vector_db, llm)
    stuff_documents_chain = create_stuff_documents_chain(llm, document_prompt)
    retriever_chain = create_retrieval_chain(retriever, stuff_documents_chain)
    return retriever_chain



with st.sidebar:
    
    if "voice_expander_label" not in st.session_state:
        st.session_state.voice_expander_label = ":green[**ON**]"
    if "enable_voice" not in st.session_state:
        st.session_state.enable_voice = True
        
    with model_settings:
        selected_model = st.selectbox("Model Name", options = llm_models, label_visibility = "collapsed", key = "selected_model")
        streaming = st.toggle(label="Streaming output", key = "streaming", value = True, help = "Enable streaming output for the assistant's response.")
        history_flag = st.toggle(label="Chat History", key="history", value = True, help = "Enable chat history or memory for the assistant's response.")
        
    if st.session_state.enable_voice == False:
        st.session_state.voice_expander_label = ":red[**OFF**]"
    if st.session_state.enable_voice == True:
        st.session_state.voice_expander_label = ":green[**ON**]"

    with st.expander(f"Voice: {st.session_state.voice_expander_label}", icon = "🎙️", expanded = True):
        enable_voice = st.toggle("Enable Voice", key = "enable_voice", value = st.session_state.enable_voice)
        if enable_voice:
            adjust_for_ambient_noise = st.toggle("Adjust for Ambient Noise", key = "adjust_for_ambient_noise", value = False, disabled = not enable_voice)
            tts_engine = st.radio("Text-to-Speech Engine", options = conf.TTS_ENGINES.keys(), index = 1, horizontal = True, disabled = not enable_voice)
            selected_voice = st.selectbox("Voice", options = conf.TTS_ENGINES[tts_engine]["voices"], disabled = not enable_voice)

    llm = init_chat_model(model = selected_model, model_provider = model_provider)
    embedding_function = OllamaEmbeddings(model = conf.EMBEDDING_MODEL)
    # embedding_function = OpenAIEmbeddings(model = conf.EMBEDDING_MODEL)

    with st.expander("Reference Document"):
        st.markdown("#### :grey[Document Collections]")
        collection_list = list_collections()
        collection = st.selectbox("Collection", options = collection_list, key = "collection", label_visibility = "collapsed", index = 0, placeholder = "Select a Collection")
        
        is_vector_db_loaded = False

        if st.session_state.collection is not None:
            st.session_state.vector_db = get_existing_vectordb(st.session_state.collection)
            chain = get_retrieval_chain(st.session_state.vector_db, llm)
            is_vector_db_loaded = True if len(st.session_state.collection) > 0 else False

        elif len(collection_list) == 0:
            st.info("⚠︎ No documents loaded. Please add new document to chat with.")

        st.toggle(
            "Use RAG",
            value = is_vector_db_loaded,
            key = "use_rag",
            disabled = not is_vector_db_loaded
        )
    
    with st.expander("Additional Settings", icon = "🔨"):
        track_token_usage = st.toggle("Track Token Usage", value = True)
        st.markdown(":grey[Debug]")
        error_traceback = st.toggle("Error Traceback")
    st.divider()
    
    st.markdown("## Instructions")
    st.markdown(":grey[To get resolution recommendations, upload your incident logs and standard operating procedure (SOP) documents, then describe your issue to the assistant.]\n\n")
    st.markdown("## About")
    st.markdown(":grey[The IT Services Incident Response Assistant is an AI-powered app that helps IT support teams quickly resolve incidents. By analyzing incident descriptions, it provides step-by-step instructions, escalation paths, and links to relevant documentation, significantly reducing resolution time.]")
    st.divider()

    st.write(f""":grey[Hostname: {socket.gethostname()}]<br>
                :grey[IP: {socket.gethostbyname(socket.gethostname())}]""", unsafe_allow_html = True)





# col1, col2 = st.columns([0.35, 0.65])
# config_container = col1.container(height = 300 if len(collection_list) > 0 else 200, border = False)

# Get Conversational RAG chain
if "voice_chat_messages" not in st.session_state:
    st.session_state.voice_chat_messages = [{
        "role": "assistant",
        "content": "Hi there! How can I help you today?",
        "audio": "",
        "avatar": utils.get_logo(selected_model),
        "input_method": "text"
    }]

if len(st.session_state.voice_chat_messages) == 1:
    st.session_state.voice_chat_messages[0]["avatar"] = utils.get_logo(selected_model)

document_message_container = st.container(height = 500, border = False)

for message in st.session_state.voice_chat_messages:
    with document_message_container:
        with st.chat_message(message["role"], avatar = message["avatar"]):
            st.markdown(message["content"], unsafe_allow_html = True)
            if message["audio"] != "":
                st.audio(message["audio"], width = 300)


input_cols = st.columns([0.95, 0.05]) if enable_voice else st.columns(1)
if enable_voice:
    voice_button = input_cols[1].button(":material/mic:", help = "Use voice mode")

############################## Text Input ##############################
user_input = {}
if user_prompt := input_cols[0].chat_input(f"Ask {st.session_state.selected_model.split(' ')[0].title()}"):
    user_input: dict[str, list] = {"role": "user", "content": user_prompt, "audio": "", "avatar": utils.get_logo("user"), "input_method": "text"}
    st.session_state.voice_chat_messages.append(user_input)
    
    with document_message_container:
        with st.chat_message(user_input["role"], avatar = utils.get_logo(user_input["avatar"])):
            st.markdown(user_input["content"], unsafe_allow_html = True)
            if user_input["audio"] != "":
                st.audio(user_input["audio"], width = 300)

############################## Voice Input ##############################
if enable_voice:
    if voice_button:
        with document_message_container:
            with st.chat_message("user", avatar = utils.get_logo("user")):
                message_placeholder = st.empty()
                try:
                    user_audio_input = utils.audio_input(adjust_for_ambient_noise = adjust_for_ambient_noise)
                    
                    with st.spinner(":grey[Transcribing...]"):
                        transcript = utils.speech_to_text(audio_data = user_audio_input)
                        user_input: dict[str, list] = {"role": "user", "content": transcript, "audio": "", "avatar": utils.get_logo("user"), "input_method": "voice"}
                        st.session_state.voice_chat_messages.append(user_input)
                except Exception as e:
                    traceback_str = str(e)
                    if error_traceback:
                        traceback_str = traceback.format_exception(e)
                    message_placeholder.markdown(f":red[Error: {traceback_str}]")
                    st.stop()
            
                message_placeholder.markdown(user_input["content"])
                if user_input["audio"] != "":
                    st.audio(user_input["audio"], width = 300)


############################## Get Response on User input ##############################
if user_input:
    with document_message_container:
        with st.chat_message("assistant", avatar = utils.get_logo(selected_model)):
            try:
                with st.spinner(":grey[Thinking...]"):
                    message_placeholder = st.empty()
                    full_response = ""
                    text_to_audio_section = ""
                    model_name = selected_model
                    
                    if st.session_state.use_rag:
                        if streaming:
                            usage_data = {}
                            if track_token_usage:
                                callback = UsageMetadataCallbackHandler()
                            record_section = False
                            ####### Stream LLM Response ########
                            for chunk in chain.pick("answer").stream({"chat_history": [{key: d[key] for key in ["role", "content"] if key in d} for d in st.session_state.voice_chat_messages[:-1]] if history_flag else [{"role": "user", "content": ""}], "input": st.session_state.voice_chat_messages[-1]["content"]}, config = {"callbacks": [callback]} if track_token_usage else None):
                            
                                # -------------- Process <think> tag for reasoning models --------------
                                if chunk == "<think>":
                                    chunk = chunk.replace("<think>", ":grey[**Reasoning**]: \n<blockquote>")
                                elif chunk == "</think>":
                                    chunk = chunk.replace("</think>", "</blockquote>")
                                    record_section = True
                                full_response += chunk
                                if record_section:
                                    if chunk not in ["<think>", "</think>", "<blockquote>", "</blockquote>"]:
                                        text_to_audio_section += chunk
                                # ----------------------------------------------------------------------
                                
                                message_placeholder.markdown(full_response + "▌", unsafe_allow_html = True)
                            if len(text_to_audio_section) == 0:
                                text_to_audio_section = full_response
                            if track_token_usage:
                                try:
                                    if len(callback.usage_metadata) != 0:
                                        usage_data = callback.usage_metadata[model_name]
                                    full_response += f":grey[  \n *Model: {model_name}  |  Input Tokens: {usage_data['input_tokens']}  |  Output Tokens: {usage_data['output_tokens']}  |  Total Tokens: {usage_data['total_tokens']}*]"
                                except Exception as e:
                                    full_response += ":grey[  \n *(Unable to extract metadata usage)*]"
                        else:
                            usage_data = {}
                            if track_token_usage:
                                callback = UsageMetadataCallbackHandler()
                                
                            ####### Invoke LLM Response ########
                            response = chain.invoke({"chat_history": [{key: d[key] for key in ["role", "content"] if key in d} for d in st.session_state.voice_chat_messages[:-1]] if history_flag else [{"role": "user", "content": ""}], "input": st.session_state.voice_chat_messages[-1]["content"]}, config = {"callbacks": [callback]} if track_token_usage else None)
                            
                            # -------------- Process <think> tag for reasoning models --------------
                            full_response = response["answer"].replace("<think>", ":grey[**Reasoning**]: \n<blockquote>").replace("</think>", "</blockquote>")
                            text_to_audio_section = full_response[full_response.find("</blockquote>")+len("</blockquoste>")-1:].strip() if full_response.find("</blockquote>") != -1 else full_response
                            # ----------------------------------------------------------------------
                            
                            if track_token_usage:
                                try:
                                    if len(callback.usage_metadata) != 0:
                                        usage_data = callback.usage_metadata[model_name]
                                    else:
                                        usage_data = response.usage_metadata
                                    full_response += f":grey[  \n *Model: {model_name}  |  Input Tokens: {usage_data['input_tokens']}  |  Output Tokens: {usage_data['output_tokens']}  |  Total Tokens: {usage_data['total_tokens']}*]"
                                except Exception as e:
                                    full_response += ":grey[  \n *(Unable to extract metadata usage)*]"
                    else:
                        if streaming:
                            usage_data = {}
                            if track_token_usage:
                                callback = UsageMetadataCallbackHandler()
                            record_section = False
                            
                            ####### Stream LLM Response ########
                            for chunk in llm.stream([{key: d[key] for key in ["role", "content"] if key in d} for d in st.session_state.voice_chat_messages] if history_flag else [{key: st.session_state.voice_chat_messages[-1][key] for key in ["role", "content"] if key in st.session_state.voice_chat_messages[-1]}], config = {"callbacks": [callback]} if track_token_usage else None):
                                
                                # -------------- Process <think> tag for reasoning models --------------
                                content = chunk.content
                                if "<think>" in content:
                                    content = content.replace("<think>", ":grey[**Reasoning**]: \n<blockquote>")
                                elif "</think>" in content:
                                    content = content.replace("</think>", "</blockquote>")
                                    record_section = True
                                full_response += content
                                if record_section:
                                    if content not in ["<think>", "</think>", "<blockquote>", "</blockquote>"]:
                                        text_to_audio_section += content
                                # ----------------------------------------------------------------------
                                
                                if "finish_reason" in chunk.response_metadata:
                                    if chunk.response_metadata["finish_reason"].lower() == "stop":
                                        usage_data = chunk.usage_metadata
                                message_placeholder.markdown(full_response + "▌", unsafe_allow_html = True)
                            if len(text_to_audio_section) == 0:
                                text_to_audio_section = full_response
                            if track_token_usage:
                                try:
                                    if len(callback.usage_metadata) != 0:
                                        usage_data = callback.usage_metadata[model_name]
                                    full_response += f":grey[  \n *Model: {model_name}  |  Input Tokens: {usage_data['input_tokens']}  |  Output Tokens: {usage_data['output_tokens']}  |  Total Tokens: {usage_data['total_tokens']}*]"
                                except Exception as e:
                                    full_response += ":grey[  \n *(Unable to extract metadata usage)*]"
                        else:
                            usage_data = {}
                            if track_token_usage:
                                callback = UsageMetadataCallbackHandler()
                            
                            ####### Invoke LLM Response ########
                            response = llm.invoke([{key: d[key] for key in ["role", "content"] if key in d} for d in st.session_state.voice_chat_messages] if history_flag else [{key: st.session_state.voice_chat_messages[-1][key] for key in ["role", "content"] if key in st.session_state.voice_chat_messages[-1]}], config = {"callbacks": [callback]} if track_token_usage else None)
                            
                            # -------------- Process <think> tag for reasoning models --------------
                            full_response = response.content.replace("<think>", ":grey[**Reasoning**]: \n<blockquote>").replace("</think>", "</blockquote>")
                            text_to_audio_section = full_response[full_response.find("</blockquote>")+len("</blockquoste>")-1:].strip() if full_response.find("</blockquote>") != -1 else full_response
                            # ----------------------------------------------------------------------
                            
                            if track_token_usage:
                                try:
                                    if len(callback.usage_metadata) != 0:
                                        usage_data = callback.usage_metadata[model_name]
                                    else:
                                        usage_data = response.usage_metadata
                                    full_response += f":grey[  \n *Model: {model_name}  |  Input Tokens: {usage_data['input_tokens']}  |  Output Tokens: {usage_data['output_tokens']}  |  Total Tokens: {usage_data['total_tokens']}*]"
                                except Exception as e:
                                    full_response += ":grey[  \n *(Unable to extract metadata usage)*]"
            except Exception as e:
                traceback_str = str(e)
                if error_traceback:
                    traceback_str = traceback.format_exception(e)
                    full_response = f":red[Error: {traceback_str}]"
                message_placeholder.markdown(full_response, unsafe_allow_html = True)
                st.session_state.voice_chat_messages.append({"role": "assistant", "content": full_response, "audio": "", "avatar": utils.get_logo(model_name)})
                st.stop()

            message_placeholder.markdown(full_response, unsafe_allow_html = True)
            
            ####### LLM Output - Text To Speech ########
            audio_bytes = None
            if enable_voice:
                try:
                    with st.spinner("Generating audio..."):
                        clean_text = text_to_audio_section.replace("#", "").strip()
                        if tts_engine == "groq":
                            audio_bytes = utils.text_to_speech_groq(clean_text, voice = selected_voice)
                        elif tts_engine == "native":
                            audio_bytes = utils.text_to_speech_native(clean_text, voice = selected_voice)
                        
                        audio_io = io.BytesIO(audio_bytes)
                        audio_io.seek(0)
                            
                    st.audio(data = audio_io, width = 300, autoplay = user_input["input_method"] == "voice")
                
                except Exception as e:
                    traceback_str = str(e)
                    if error_traceback:
                        traceback_str = traceback.format_exception(e)
                    st.markdown(f":red[Error: {traceback_str}]")
            
            st.session_state.voice_chat_messages.append({"role": "assistant", "content": full_response, "audio": audio_io, "avatar": utils.get_logo(model_name)})
