import streamlit as st
import asyncio
import aiohttp
from datetime import datetime

# --- CONFIG ---
USER_BG = "linear-gradient(135deg, #90ee90, #00ff7f)"
ASSISTANT_BG = "linear-gradient(135deg, #e0e0e0, #f5f5f5)"
USER_AVATAR = "🧑"
ASSISTANT_AVATAR = "🤖"

CARD_STYLE = """
background-color:#f7f7f7;
padding:15px;
border-radius:12px;
box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
margin-bottom:10px;
"""

BUTTON_STYLE = """
<style>
.add-docs-button>button {
    background: linear-gradient(90deg, #FF8C00, #FFA500);
    color: white;
    font-weight: bold;
    border-radius: 8px;
    height: 35px;
    width: 100%;
}
.add-docs-button>button:hover {
    background: linear-gradient(90deg, #FFA500, #FF8C00);
}
</style>
"""


def main():
    st.set_page_config(page_title="Smart Assistant", page_icon="🤖", layout="wide")

    # Title
    st.markdown("<h1 style='text-align:center; color:#4B0082;'>Smart Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-style:italic;'>Powered by ChromaDB & Tavily</p>", unsafe_allow_html=True)

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "backend_url" not in st.session_state:
        st.session_state.backend_url = "http://localhost:8000"

    # --- Sidebar ---
    with st.sidebar:
        # Settings card
        st.markdown(f"<div style='{CARD_STYLE}'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#4B0082;'>Settings</h4>", unsafe_allow_html=True)
        st.session_state.backend_url = st.text_input("Backend URL", st.session_state.backend_url)
        st.markdown("</div>", unsafe_allow_html=True)

        # Upload documents card
        st.markdown(f"<div style='{CARD_STYLE}'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#4B0082;'>Upload Documents</h4>", unsafe_allow_html=True)
        files = st.file_uploader("Select files", type=["txt","pdf","docx"], accept_multiple_files=True)
        st.markdown(BUTTON_STYLE.replace("host-button", "add-docs-button"), unsafe_allow_html=True)
        if files and st.button("Add to Knowledge Base", key="add_docs"):
            with st.spinner("Uploading..."):
                asyncio.run(upload_documents(files))
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Chat container ---
    chat_container = st.container()
    for msg in st.session_state.messages:
        render_message(msg, chat_container)

    if prompt := st.chat_input("Type your message..."):
        add_message("user", prompt)
        render_message(st.session_state.messages[-1], chat_container)
        asyncio.run(handle_response(prompt, chat_container))

def render_message(msg, container=None):
    role = msg["role"]
    content = msg["content"]
    timestamp = msg.get("time", datetime.now().strftime("%H:%M"))

    bg = USER_BG if role=="user" else ASSISTANT_BG
    avatar = USER_AVATAR if role=="user" else ASSISTANT_AVATAR
    align = "right" if role=="user" else "left"

    markdown = f"""
    <div style='display:flex; justify-content:{align}; margin:5px 0;'>
        <div style='max-width:70%; display:flex; align-items:flex-end; flex-direction:{'row-reverse' if role=='user' else 'row'}'>
            <div style='font-size:24px; margin:0 5px'>{avatar}</div>
            <div style='background:{bg}; padding:10px; border-radius:12px; word-wrap:break-word;'>
                {content}<br><span style='font-size:0.7em; color:gray'>{timestamp}</span>
            </div>
        </div>
    </div>
    """
    if container:
        container.markdown(markdown, unsafe_allow_html=True)
    else:
        st.markdown(markdown, unsafe_allow_html=True)

def add_message(role, content):
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "time": datetime.now().strftime("%H:%M")
    })

async def handle_response(prompt, container):
    add_message("assistant", "Thinking...")
    render_message(st.session_state.messages[-1], container)
    
    response = await get_agent_response(prompt)
    st.session_state.messages[-1]["content"] = response

    # Update chat
    container.empty()
    for msg in st.session_state.messages:
        render_message(msg, container)

async def upload_documents(files):
    try:
        async with aiohttp.ClientSession() as session:
            for f in files:
                data = aiohttp.FormData()
                data.add_field("file", f.getvalue(), filename=f.name)
                async with session.post(f"{st.session_state.backend_url}/upload", data=data) as resp:
                    if resp.status == 200:
                        st.success(f"Uploaded {f.name}")
                    else:
                        st.error(f"Failed {f.name}")
    except Exception as e:
        st.error(f"Upload error: {e}")

async def get_agent_response(query):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{st.session_state.backend_url}/chat", json={"query": query}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("response", "")
                else:
                    return "Backend error."
    except Exception as e:
        return f"Connection error: {e}"

if __name__ == "__main__":
    main()



