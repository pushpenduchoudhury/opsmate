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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


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
    st.button(":material/mop:", on_click = lambda: clear_messages(), type = "secondary", width = "stretch")

    
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
        tts_engine = st.radio("Text-to-Speech Engine", options = conf.TTS_ENGINES.keys(), index = 1, horizontal = True, disabled = not enable_voice)
        selected_voice = st.selectbox("Voice", options = conf.TTS_ENGINES[tts_engine]["voices"], disabled = not enable_voice)


    st.divider()
    
    st.markdown("## Instructions")
    st.markdown(":grey[To get resolution recommendations, upload your incident logs and standard operating procedure (SOP) documents, then describe your issue to the assistant.]\n\n")
    st.markdown("## About")
    st.markdown(":grey[The IT Services Incident Response Assistant is an AI-powered app that helps IT support teams quickly resolve incidents. By analyzing incident descriptions, it provides step-by-step instructions, escalation paths, and links to relevant documentation, significantly reducing resolution time.]")
    st.divider()

    st.write(f""":grey[Hostname: {socket.gethostname()}]<br>
                :grey[IP: {socket.gethostbyname(socket.gethostname())}]""", unsafe_allow_html = True)

llm = init_chat_model(model = selected_model, model_provider = model_provider)
embedding_function = OllamaEmbeddings(model = conf.EMBEDDING_MODEL)
# embedding_function = OpenAIEmbeddings(model = conf.EMBEDDING_MODEL)

def document_loader(document_files: list):
    """Load a document file and return its content."""
    documents: list = []

    # Create or set a temporary directory to store the uploaded files
    os.makedirs(conf.TEMP_DIR, exist_ok = True)

    for file in document_files:
        temp_file_path = Path(conf.TEMP_DIR, file.name)
        file_extension: str = file.name.split(".")[-1].lower()
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file.read())

        try:
            if file_extension == "pdf":
                loader = PyPDFLoader(temp_file_path)
            elif file_extension == "docx":
                loader = Docx2txtLoader(temp_file_path)
            elif file_extension in ["txt", "md", "py"]:
                loader = TextLoader(temp_file_path, encoding = "utf-8")
            elif file_extension == "xlsx":
                loader = UnstructuredExcelLoader(temp_file_path)
            elif file_extension == "pptx":
                loader = UnstructuredPowerPointLoader(temp_file_path)
        except Exception as e:
            st.error(f"Error loading {file.name}: {e}", icon = "⚠️")
            st.stop()
        finally:
            documents.extend(loader.load())
            os.remove(temp_file_path)
    return documents

def url_loader(url):
    documents: list = []
    loader = WebBaseLoader(url)
    loader.requests_kwargs = {'verify':False}
    documents.extend(loader.load())
    return documents
    
def split_documents(documents) -> list[str]:
    """ Load a text file and split it into chunks."""
    chunk_size = conf.CHUNK_SIZE
    chunk_overlap = conf.CHUNK_OVERLAP
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vectordb(documents: list, collection_name) -> Chroma:
    """Create a vector database from the documents using Google Generative AI embeddings."""
    vectordb = Chroma.from_documents(documents, embedding = embedding_function, collection_name = collection_name, persist_directory = str(conf.CHROMADB_DIR))
    return vectordb

def get_existing_vectordb(collection_name) -> Chroma:
    client = chromadb.Client(Settings(is_persistent = True, persist_directory = str(conf.CHROMADB_DIR)))
    vectordb = Chroma(client = client, collection_name = collection_name, embedding_function = embedding_function)
    return vectordb

def list_collections():
    client = chromadb.Client(Settings(is_persistent = True, persist_directory = str(conf.CHROMADB_DIR)))
    collections = [collection.name for collection in client.list_collections()]
    return collections

def delete_collection(collection_name):
    client = chromadb.Client(Settings(is_persistent = True, persist_directory = str(conf.CHROMADB_DIR)))
    client.delete_collection(name = collection_name)

def get_embedded_documents(vector_db: Chroma) -> list:
    metadata: list[dict] = vector_db.get()["metadatas"]
    file_list = []
    for file in metadata:
        if str(file['source']).startswith("http"):
            file_list.append(file["source"])
        else:
            file_list.append(Path(file["source"]).name)
    embedded_files = list(set(file_list))
    return embedded_files

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
            ("system", """You are a helpful assistant. You will have to answer to user's prompts.
                        You will have some context to help with your answers, but it might not always would be completely related or helpful. Do not quote references from the document provided, respond in your own language.
                        You can also use your knowledge to assist answering the user's propmts.\n
            {context}"""),
            MessagesPlaceholder(variable_name = "chat_history"),
            ("user", "{input}"),
        ])
    retriever = get_history_aware_retriever(vector_db, llm)
    stuff_documents_chain = create_stuff_documents_chain(llm, document_prompt)
    retriever_chain = create_retrieval_chain(retriever, stuff_documents_chain)
    return retriever_chain

if "embedded_documents" not in st.session_state:
    st.session_state.embedded_documents = []
    
def load_rag_sources(collection_name, source_type):
    with st.spinner("Embedding Document..."):
        
        if source_type == "Documents":
            if st.session_state.selected_files:                
                documents = document_loader(st.session_state.selected_files)
        elif source_type == "URL":
            if st.session_state.selected_url:
                documents = url_loader(st.session_state.selected_url)
        
        # Create chunks from the documents
        chunks = split_documents(documents)
        if len(chunks) > 0:
            try:
                create_vectordb(chunks, collection_name)
                if source_type == "Documents":
                    st.toast(f"Document *{str([file.name for file in st.session_state.selected_files])[1:-1]}* loaded successfully", icon = "✅")
                elif source_type == "URL":
                    st.toast(f"URL *{str(st.session_state.selected_url)}* loaded successfully", icon = "✅")

            except Exception as e:
                traceback_str = traceback.format_exception(e)
                st.error(traceback_str)
                st.stop()
        else:
            st.toast(f"ⓘ Document *{str([file.name for file in st.session_state.selected_files])[1:-1] if source_type == 'Documents' else st.session_state.selected_url}* already loaded in Collection")
            del st.session_state.show_add_document

if "show_add_document" not in st.session_state:
    st.session_state.show_add_document = False

def show_add_document():
    st.session_state.show_add_document = True

def hide_add_document():
    st.session_state.show_add_document = False


col1, col2 = st.columns([0.35, 0.65])

col1.markdown("#### :blue[Document Collections]")

collection_list = list_collections()

if "admin" in st.session_state.get("roles"):
    cols1 = col1.columns([0.6, 0.1, 0.1])
    collection = cols1[0].selectbox("Collection", options = collection_list, key = "collection", label_visibility = "collapsed", index = 0, placeholder = "Select a Collection")
    cols1[1].button(":material/add:", on_click = show_add_document, help = "Add Document to Selected Collection", width = "stretch")
    cols1[2].button(":material/delete:", type = "primary", on_click = lambda: delete_collection(collection_name = st.session_state.collection), help = "Delete Selected Collection", disabled = True if collection is None else False, width = "stretch")
else:
    collection = col1.selectbox("Collection", options = collection_list, key = "collection", label_visibility = "collapsed", index = 0, placeholder = "Select a Collection")
    

is_vector_db_loaded = False

if st.session_state.collection is not None:
    st.session_state.vector_db = get_existing_vectordb(st.session_state.collection)
    chain = get_retrieval_chain(st.session_state.vector_db, llm)
    is_vector_db_loaded = True if len(st.session_state.collection) > 0 else False

elif len(collection_list) == 0:
    col1.info("⚠︎ No documents loaded. Please add new document to chat with.")

col1.toggle(
    "Use RAG",
    value = is_vector_db_loaded,
    key = "use_rag",
    disabled = not is_vector_db_loaded
)

config_container = col1.container(height = 300 if len(collection_list) > 0 else 200, border = False)

if st.session_state.collection is not None:
    st.session_state.embedded_docs = get_embedded_documents(get_existing_vectordb(collection_name = st.session_state.collection))
    with config_container.expander(label = f":grey[Documents in Collection ({len(st.session_state.embedded_docs)})]", expanded = False):
        st.write(st.session_state.embedded_docs)

if st.session_state.show_add_document:
    col3, col4, col5 = config_container.columns([0.7, 0.1, 0.01])
    col3.markdown("#### :blue[Add or Create RAG Sources]")
    col4.button(":material/close:", type = "tertiary", on_click = hide_add_document, help = "Close", width = "stretch")
    new_collection_name = config_container.text_input("New Collection Name", value = None, placeholder = "New Collection Name", label_visibility = "collapsed")
    source_type = config_container.radio("Source Type", options = ["Documents", "URL"], horizontal = True, label_visibility = "collapsed")

    if source_type == "Documents":
        uploaded_flg = False
        config_container.file_uploader(
            "🗎 Upload a document",
            type = ["pdf", "txt", "docx", "md", "xlsx", "pptx"],
            accept_multiple_files = True,
            key = "selected_files",
            help = "Upload a document to chat with it.",
            label_visibility = "collapsed"
        )
        if len(collection_list) == 0:
            if len(st.session_state.selected_files) > 0 and new_collection_name is not None and new_collection_name != "":
                uploaded_flg = True
        else:
            if len(st.session_state.selected_files) > 0:
                uploaded_flg = True
        
        config_container.button(f"Add Files to collection: '{st.session_state.collection}'" if new_collection_name is None or new_collection_name == "" else f"Create new collection: '{new_collection_name.replace(' ', '_')}'", on_click = lambda: load_rag_sources(collection_name = st.session_state.collection if new_collection_name is None or new_collection_name == "" else new_collection_name.replace(' ', '_'), source_type = "Documents"), disabled = not uploaded_flg)
        
    if source_type == "URL":
        uploaded_flg = False
        config_container.text_input("🌐︎ Provide a URL", key = "selected_url", placeholder = "https://example.com", label_visibility = "collapsed")

        if len(collection_list) == 0:
            if len(st.session_state.selected_url) > 0 and new_collection_name is not None and new_collection_name != "":
                uploaded_flg = True
        else:
            if len(st.session_state.selected_url) > 0:
                uploaded_flg = True
                
        config_container.button(f"Add URL to collection: '{st.session_state.collection}'" if new_collection_name is None or new_collection_name == "" else f"Create new collection: '{new_collection_name.replace(' ', '_')}'", on_click = lambda: load_rag_sources(collection_name = st.session_state.collection if new_collection_name is None or new_collection_name == "" else new_collection_name.replace(' ', '_'), source_type = "URL"), disabled = not uploaded_flg)

# Get Conversational RAG chain
if "voice_chat_messages" not in st.session_state:
    st.session_state.voice_chat_messages = [{
        "role": "assistant",
        "content": "Hi there! How can I help you today?",
        "audio": ""
    }]

document_message_container = col2.container(height = 450, border = True)

for message in st.session_state.voice_chat_messages:
    with document_message_container:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["audio"] != "":
                st.audio(message["audio"], width = 300)


input_cols = st.columns([0.95, 0.05]) if enable_voice else st.columns(1)
if enable_voice:
    voice_button = input_cols[1].button(":material/mic:", help = "Use voice mode")

if user_prompt := input_cols[0].chat_input(f"Ask {st.session_state.selected_model.split(' ')[0].title()}"):
    user_input: dict[str, list] = {"role": "user", "content": user_prompt, "audio": ""}
    st.session_state.voice_chat_messages.append(user_input)

    with document_message_container:
        with st.chat_message(user_input["role"]):
            st.markdown(user_input["content"])
            if user_input["audio"] != "":
                st.audio(user_input["audio"], width = 300)

        with st.chat_message("assistant"):
            try:
                with st.spinner(":grey[Thinking...]"):
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    if st.session_state.use_rag:
                        if streaming:
                            for chunk in chain.pick("answer").stream({"chat_history": [{key: d[key] for key in ["role", "content"] if key in d} for d in st.session_state.voice_chat_messages[:-1]] if history_flag else [{"role": "user", "content": "", "audio": ""}], "input": st.session_state.voice_chat_messages[-1]["content"]}):
                                full_response += chunk
                                message_placeholder.markdown(full_response + "▌")
                        else:
                            response = chain.invoke({"chat_history": [{key: d[key] for key in ["role", "content"] if key in d} for d in st.session_state.voice_chat_messages[:-1]] if history_flag else [{"role": "user", "content": "", "audio": ""}], "input": st.session_state.voice_chat_messages[-1]["content"]})
                            full_response = response["answer"]
                    else:
                        if streaming:
                            for chunk in llm.stream([{key: d[key] for key in ["role", "content"] if key in d} for d in st.session_state.voice_chat_messages] if history_flag else {key: st.session_state.voice_chat_messages[-1][key] for key in ["role", "content"] if key in st.session_state.voice_chat_messages[-1]}):
                                full_response += chunk.content
                                message_placeholder.markdown(full_response + "▌")
                        else:
                            response = llm.invoke([{key: d[key] for key in ["role", "content"] if key in d} for d in st.session_state.voice_chat_messages] if history_flag else {key: st.session_state.voice_chat_messages[-1][key] for key in ["role", "content"] if key in st.session_state.voice_chat_messages[-1]})
                            full_response = response.content
                    
            except Exception as e:
                full_response = f":red[Error: {str(e)}]"
                message_placeholder.markdown(full_response)
                st.session_state.voice_chat_messages.append({"role": "assistant", "content": full_response, "audio": ""})
                st.stop()

            message_placeholder.markdown(full_response)
            
            audio = ""
            if enable_voice:
                try:
                    if tts_engine == "groq":
                        audio = utils.text_to_speech_groq(full_response, voice = selected_voice)
                    elif tts_engine == "native":
                        audio = utils.text_to_speech_native(full_response, voice = selected_voice)
                    st.audio(audio, width = 300)
                except Exception as e:
                    st.markdown(f":red[Error: {str(e)}]")
            
            st.session_state.voice_chat_messages.append({"role": "assistant", "content": full_response, "audio": audio})

if enable_voice:
    if voice_button:
        with document_message_container:
            with st.chat_message("user"):
                message_placeholder = st.empty()
                try:
                    with st.spinner(":grey[Listening...]"):
                        user_audio_input = utils.audio_input()
                    
                    with st.spinner(":grey[Transcribing...]"):
                        transcript = utils.speech_to_text(audio_data = user_audio_input)
                        user_input: dict[str, list] = {"role": "user", "content": transcript, "audio": ""}
                        st.session_state.voice_chat_messages.append(user_input)
                except Exception as e:
                    message_placeholder.markdown(f":red[Error: {str(e)}]")
                    st.stop()
                
                st.markdown(user_input["content"])
                if user_input["audio"] != "":
                    st.audio(user_input["audio"], width = 300)
            
            with st.chat_message("assistant"):
                try:
                    with st.spinner(":grey[Thinking...]"):
                        message_placeholder = st.empty()
                        full_response = ""
                        
                        if st.session_state.use_rag:
                            if streaming:
                                for chunk in chain.pick("answer").stream({"chat_history": [{key: d[key] for key in ["role", "content"] if key in d} for d in st.session_state.voice_chat_messages[:-1]] if history_flag else [{"role": "user", "content": "", "audio": ""}], "input": st.session_state.voice_chat_messages[-1]["content"]}):
                                    full_response += chunk
                                    message_placeholder.markdown(full_response + "▌")
                            else:
                                response = chain.invoke({"chat_history": [{key: d[key] for key in ["role", "content"] if key in d} for d in st.session_state.voice_chat_messages[:-1]] if history_flag else [{"role": "user", "content": "", "audio": ""}], "input": st.session_state.voice_chat_messages[-1]["content"]})
                                full_response = response["answer"]
                        else:
                            if streaming:
                                for chunk in llm.stream([{key: d[key] for key in ["role", "content"] if key in d} for d in st.session_state.voice_chat_messages] if history_flag else [{key: st.session_state.voice_chat_messages[-1][key] for key in ["role", "content"] if key in st.session_state.voice_chat_messages[-1]}]):
                                    full_response += chunk.content
                                    message_placeholder.markdown(full_response + "▌")
                            else:
                                response = llm.invoke([{key: d[key] for key in ["role", "content"] if key in d} for d in st.session_state.voice_chat_messages] if history_flag else [{key: st.session_state.voice_chat_messages[-1][key] for key in ["role", "content"] if key in st.session_state.voice_chat_messages[-1]}])
                                full_response = response.content
                            
                except Exception as e:
                    full_response = f":red[Error: {str(e)}]"
                    message_placeholder.markdown(full_response)
                    st.session_state.voice_chat_messages.append({"role": "assistant", "content": full_response, "audio": ""})
                    st.stop()

                message_placeholder.markdown(full_response)
                
                audio = ""
                if enable_voice:
                    try:
                        if tts_engine == "groq":
                            audio = utils.text_to_speech_groq(full_response, voice = selected_voice)
                        elif tts_engine == "native":
                            audio = utils.text_to_speech_native(full_response, voice = selected_voice)
                        st.audio(audio, autoplay = True, width = 300)
                    except Exception as e:
                        st.markdown(f":red[Error: {str(e)}]")
                
                st.session_state.voice_chat_messages.append({"role": "assistant", "content": full_response, "audio": audio})
