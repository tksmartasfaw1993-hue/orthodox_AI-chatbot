import os
import streamlit as st
import ollama
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, WebBaseLoader, YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# 1. መሠረታዊ ውቅሮች (Configurations)
DB_DIR = "./orthodox_knowledge_db"
BOOKS_DIR = "./other"

# የStreamlit ገጽ ርዕስ
st.set_page_config(page_title="Orthodox AI Chatbot", page_icon="⛪")
st.title("⛪ የኦርቶዶክስ ተዋሕዶ ረዳት AI")

# 2. ሁሉንም መረጃዎች (PDF, Web, YouTube) ሰብስቦ ማሰልጠኛ ፈንክሽን
def train_and_build_db():
    st.info("⏳ የኦርቶዶክስ መጻሕፍት፣ ዌብሳይት እና ዩቲዩብ መረጃዎችን የመጫን ስራ ተጀምሯል...")
    all_documents = []

    if os.path.exists(BOOKS_DIR):
        loader = DirectoryLoader(BOOKS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
        all_documents.extend(loader.load())
    else:
        os.makedirs(BOOKS_DIR)

    web_articles = ["https://debelo.org", "https://eotcmk.org", "https://eotc.org.et"]
    for web_url in web_articles:
        try:
            loader = WebBaseLoader(web_url)
            all_documents.extend(loader.load())
        except:
            pass

    youtube_videos = ["https://youtube.com", "https://youtube.com"]
    for video_url in youtube_videos:
        try:
            loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=True, language=["am", "en"])
            all_documents.extend(loader.load())
        except:
            pass

    if not all_documents:
        st.error("❌ ምንም መረጃ አልተገኘም። እባክዎ መጽሐፍ ወይም ኢንተርኔት ያረጋግጡ።")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    docs = text_splitter.split_documents(all_documents)

    embeddings = OllamaEmbeddings(model="llama3")
    vector_db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=DB_DIR)
    st.success("🎉 መረጃዎቹን በሙሉ የማጥናት ስራ በተሳካ ሁኔታ ተጠናቋል!")
    return vector_db

# 3. የAI መልስ መፈለጊያ ፈንክሽን
def get_ai_response(user_query: str):
    embeddings = OllamaEmbeddings(model="llama3")
    if os.path.exists(DB_DIR):
        vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    else:
        vector_db = train_and_build_db()
        if not vector_db:
            return "❌ ይቅርታ፣ የእውቀት ዳታቤዙ ገና አልተሰራም።"

    results = vector_db.similarity_search(user_query, k=3)
    context = "\n".join([doc.page_content for doc in results])
    sources = "\n".join([f"- {doc.metadata.get('source', 'የአካባቢ መጽሐፍ/ድረ-ገጽ')}" for doc in results])
    
    system_prompt = f"""
    አንተ ታማኝ የኢትዮጵያ ኦርቶዶክስ ተዋሕዶ ቤተክርስቲያን ረዳት AI ነህ።
    ከታች ከተሰጠህ መረጃ (Context) ውጪ ምንም አይነት የራስህን ሃሳብ ወይም ታሪክ አትፍጠር።
    መልሱ በመረጃው ውስጥ ከሌለ በቅንነት 'ይቅርታ፣ ይህ መረጃ እኔ ካለኝ መጻሕፍት ውስጥ አልተገኘም' በል።
    ሁልጊዜ በአማርኛ ቋንቋ ብቻ መልስ።
    
    መረጃው (Context):
    {context}
    """
    try:
        response = ollama.chat(model='llama3', messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_query}
        ])
        bot_reply = response['message']['content']
        return f"{bot_reply}\n\n📍 [የመረጃው ምንጭ]፦\n{sources}"
    except Exception as e:
        return f"❌ ስህተት ተፈጥሯል፦ {e} \n\n(ማሳሰቢያ፦ ይህ አፕ በኦንላይን ሰርቨር ላይ በሚሆንበት ጊዜ ከኮምፒውተርዎ Ollama ጋር ለመገናኘት የ Ngrok ሊንክ ያስፈልገዋል)"

# 4. የStreamlit ቻት ኢንተርፌስ (Chat UI)
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("ጥያቄዎን እዚህ ይጻፉ..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("በመፈለግ ላይ..."):
            reply = get_ai_response(user_input)
            st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
