import streamlit as st

from chatbot import generate_answer
from utils import *


st.set_page_config(page_title="Chatbot Playlist", page_icon="🔗", layout="centered")

if "link_salvo" not in st.session_state:
    st.session_state.link_salvo = None
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

st.markdown(
    """
    <style>
    .card {border: 1px solid #ddd; border-radius:8px; padding:1rem;
           background:#fff; box-shadow:0 2px 4px rgba(0,0,0,0.1); margin-bottom:1rem;}
    .card input {width:100%; padding:0.5rem; border:1px solid #ccc;
                 border-radius:4px; margin-bottom:0.5rem;}
    .card button {width:100%; padding:0.5rem; border:none; border-radius:4px;
                  background:#4CAF50; color:white; font-weight:600;}
    .playlist-link {font-size:0.95rem; color:#333; word-break:break-all;}
    </style>
    """,
    unsafe_allow_html=True,
)

pagina = st.sidebar.radio("", ["💬 Chatbot", "🔗 Gerenciar Playlist"], label_visibility="collapsed")

if pagina == "💬 Chatbot":
    st.title("🤖 Chatbot Playlist")
    if not st.session_state.link_salvo:
        st.info("Adicione uma playlist para iniciar o chat.", icon="💡")
    else:
        for autor, msg in st.session_state.mensagens:
            tipo = "user" if autor == "🧑 Você" else "assistant"
            with st.chat_message(tipo):
                st.markdown(msg)
        with st.chat_message("user"):
            with st.form("form_chat", clear_on_submit=True):
                entrada = st.text_input("", placeholder="Digite sua mensagem...", label_visibility="collapsed")
                if st.form_submit_button("Enviar"):
                    entrada = sanitize_prompt(entrada)
                    st.session_state.mensagens.append(("🧑 Você", entrada))
                    resposta = generate_answer(st.session_state.link_salvo, entrada)
                    st.session_state.mensagens.append(("🤖 Bot", resposta))
                    st.rerun()

elif pagina == "🔗 Gerenciar Playlist":
    st.title("🔗 Gerenciar Playlist")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if st.session_state.link_salvo:
        st.markdown(
            f'<div class="playlist-link">Playlist ativa:<br>'
            f'<a href="{st.session_state.link_salvo}" target="_blank">'
            f'{st.session_state.link_salvo}</a></div>',
            unsafe_allow_html=True
        )
        if st.button("Remover playlist", use_container_width=True):
            st.session_state.link_salvo = None
            st.session_state.mensagens = []
            st.success("Playlist removida com sucesso!", icon="❌")
    else:
        with st.form("form_link", clear_on_submit=True):
            link = st.text_input("", placeholder="Cole o link da playlist aqui", label_visibility="collapsed")
            if st.form_submit_button("Salvar"):
                if link.strip():
                    st.session_state.link_salvo = link
                    create_pc_playlist(link)
                    add_video_pc(link)
                    st.session_state.mensagens = []
                    st.success("Playlist salva com sucesso!", icon="✅")
    st.markdown('</div>', unsafe_allow_html=True)
