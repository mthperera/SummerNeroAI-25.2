import os
import re
from enum import Enum
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents.base import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel, Field
from pytube import Playlist, YouTube
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp

load_dotenv(override=True)


def get_videos_links(url_playlist: str) -> list[str]:

    url = url_playlist
    playlist = Playlist(url)
    urls_videos = list(playlist.video_urls)
    return urls_videos


def get_video_transcription(urls_videos: list[str]) -> list[str]:
    
    transcriptions = list()
    for url_video in urls_videos:
        try:
            video_id = url_video.find("v=")
            video_id = url_video[video_id+2:]
            ytt_api = YouTubeTranscriptApi() 
            transcript = ytt_api.get_transcript(video_id)
            transcription = ""
            for transc in transcript:
                transcription += transc["text"] + " "
            transcriptions.append(transcription)
        except:
            transcriptions.append("Sem transcrição disponível")
    return transcriptions


def get_videos_metadata(urls_videos: list[str]) -> list[dict]:

    transcriptions = get_video_transcription(urls_videos)
    lista_videos_medata = list()
    for i, url_video in enumerate(urls_videos):
        try:
            video = YouTube(url_video)
            metadata_video = {
                "Url Vídeo":                url_video,
                "Título":                   video.title,
                "Descrição":                video.description,
                "Data de Publicação":       video.publish_date.strftime('%Y-%m-%d'),
                "Duração (segundos)":       video.length,
                "Canal":                    video.author,
                "URL do Canal":             video.channel_url,
                "Visualizações":            video.views,
                "Palavras-chave":           video.keywords,
                "Thumbnail":                video.thumbnail_url,
                "Transcrição":              transcriptions[i]
            }
            lista_videos_medata.append(metadata_video)
        except Exception as e:
            print(f"Tentando com yt-dlp. {e}")
            try:
                ydl_opts = {
                    'quiet': True,
                    'skip_download': True,
                    'forcejson': True
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url_video, download=False)
                    metadata_video = {
                        "Url Vídeo":                url_video,
                        "Título":                   info.get("title"),
                        "Descrição":                info.get("description"),
                        "Data de Publicação":       info.get("upload_date"),
                        "Duração (segundos)":       info.get("duration"),
                        "Canal":                    info.get("uploader"),
                        "URL do Canal":             info.get("channel_url"),
                        "Visualizações":            info.get("view_count"),
                        "Palavras-chave":           info.get("tags"),
                        "Thumbnail":                info.get("thumbnail"),
                        "Transcrição":              transcriptions[i]
                    }
                    lista_videos_medata.append(metadata_video)
                    print("Conseguimos com o yt-dlp")
            except Exception as e:
                print(f"yt-dlp também falhou. {e}")
                metadata_video = {
                    "Url Vídeo":                url_video,
                    "Título":                   "Indisponível",
                    "Descrição":                "Indisponível",
                    "Data de Publicação":       "Indisponível",
                    "Duração (segundos)":       "Indisponível",
                    "Canal":                    "Indisponível",
                    "URL do Canal":             "Indisponível",
                    "Visualizações":            "Indisponível",
                    "Palavras-chave":           "Indisponível",
                    "Thumbnail":                "Indisponível",
                    "Transcrição":              transcriptions[i]
                }
                lista_videos_medata.append(metadata_video)
    return lista_videos_medata


def sanitize_index_name(url: str) -> str:
    query = parse_qs(urlparse(url).query)
    playlist_id = query.get("list", [""])[0]
    name = playlist_id.lower()
    name = re.sub(r'[^a-z0-9\-]', '-', name)
    name = re.sub(r'-+', '-', name)
    name = name.strip('-')
    return name


def create_pc_playlist(url_playlist: str):

    pc = Pinecone(os.getenv("PINECONE_API_KEY"))
    index_name = sanitize_index_name(url_playlist)
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )


def get_pc_playlist(url_playlist: str) -> PineconeVectorStore:

    embeddings = OpenAIEmbeddings(
        model = "text-embedding-3-small",
        dimensions = 1536
    )
    pc = Pinecone()
    index_name = sanitize_index_name(url_playlist)
    pc_index = pc.Index(index_name)
    vector_store = PineconeVectorStore(
        embedding=embeddings,
        index=pc_index
    )
    return {"vector_store": vector_store, "pc_index": pc_index}


def get_vdb_chat(url_playlist: str) -> Chroma:

    embeddings = OpenAIEmbeddings(
        model = "text-embedding-3-small",
        dimensions = 1536
    )
    index_name = sanitize_index_name(url_playlist)
    vector_store = Chroma(
    collection_name=index_name,
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
    )
    return vector_store


def add_video_pc(url_playlist: str):

    urls_videos = get_videos_links(url_playlist)
    stats = get_pc_playlist(url_playlist)["pc_index"].describe_index_stats()
    if stats["total_vector_count"] == 0:
        vector_store = get_pc_playlist(url_playlist)["vector_store"]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200
        )
        lista_videos_metadata = get_videos_metadata(urls_videos)
        for video in lista_videos_metadata:
            chunks_video = text_splitter.split_text(video["Transcrição"])
            lista_documents = list()
            for chunk in chunks_video:
                chunk_document = Document(
                    page_content = chunk
                )
                chunk_document.metadata["Url Vídeo"]            = video["Url Vídeo"]
                chunk_document.metadata["Título"]               = video["Título"]
                chunk_document.metadata["Descrição"]            = video["Descrição"]
                chunk_document.metadata["Data de Publicação"]   = video["Data de Publicação"]
                chunk_document.metadata["Duração (segundos)"]   = video["Duração (segundos)"]
                chunk_document.metadata["Canal"]                = video["Canal"]
                chunk_document.metadata["URL do Canal"]         = video["URL do Canal"]
                chunk_document.metadata["Visualizações"]        = video["Visualizações"]
                chunk_document.metadata["Palavras-chave"]       = video["Palavras-chave"]
                chunk_document.metadata["Thumbnail"]            = video["Thumbnail"]
                lista_documents.append(chunk_document)
            vector_store.add_documents(lista_documents)


def sanitize_prompt(prompt: str) -> str:

    idx_abre = prompt.find('{')
    if idx_abre != -1:
        idx_fecha = prompt.find('}', idx_abre + 1)
        if idx_fecha != -1:
            return "Eu sou um frouxo"

    return prompt.replace('{', '').replace('}', '')


class GuardElement(Enum):

    SIM = "SIM"
    NAO = "NAO"

class GuardRail(BaseModel):

    binary_score : GuardElement = Field(description="A mensagem deve ser bloqueada: 'SIM ou 'NAO'")