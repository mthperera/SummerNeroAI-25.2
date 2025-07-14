from operator import itemgetter

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai.chat_models.base import ChatOpenAI

from utils import *

load_dotenv(override=True)

def generate_answer(url_playlist: str, question_user: str) -> str:

    model = ChatOpenAI(model="gpt-4o")

    vector_store_pc = get_pc_playlist(url_playlist)["vector_store"]
    results_pc = vector_store_pc.similarity_search(query=question_user, k=3)
    string_result_pc = str()
    for i, res in enumerate(results_pc):
        string_result_pc += f"""
        Trecho do vídeo {i}:
        {res.page_content}

        Informações do vídeo {i}:
        - Título: {res.metadata["Título"]}
        - Descrição: {res.metadata["Descrição"]}
        - Data de Publicação: {res.metadata["Data de Publicação"]}
        - Duração (segundos): {res.metadata["Duração (segundos)"]}
        - Canal: {res.metadata["Canal"]}
        - URL do Canal: {res.metadata["URL do Canal"]}
        - Visualizações: {res.metadata["Visualizações"]}
        - Palavras-chave: {res.metadata["Palavras-chave"]}
        - Thumbnail: {res.metadata["Thumbnail"]}
        - Url do vídeo: {res.metadata["Url Vídeo"]}
        """
        string_result_pc += "\n\n\n"
    
    vector_store_chroma = get_vdb_chat(url_playlist)
    results_chroma = vector_store_chroma.similarity_search(query=question_user, k=4)
    string_result_chroma = str()
    for res in results_chroma:
        string_result_chroma += res.metadata["author"] + " " + res.page_content + "\n\n"

    
    messages = [
        ('system', 
        "Você é um assistente especializado em vídeos de estudo. Sua função é conversar exclusivamente sobre os vídeos presentes em uma playlist específica. "
        "Você deve responder apenas com base nas informações que possui sobre esses vídeos. "
        "Fale sobre o conteúdo do vídeo, os temas abordados, o estilo de explicação, o apresentador, ou qualquer outro detalhe relevante. "
        "Se o vídeo mencionado não estiver na playlist conhecida, responda educadamente que você só pode conversar sobre os vídeos da playlist disponível. "
        "Não responda perguntas fora do contexto dos vídeos da playlist. "
        "Seja engajado e amigável, incentivando o usuário a explorar e conversar mais sobre os vídeos que ele gostou ou quer entender melhor."),
        
        ('system', 
        "O contexto dos vídeos é:\n{context}\n"
        "Responda com base no vídeo cujo conteúdo mais se encaixa com a pergunta ou interesse do usuário."),
        ('user',"""
        Responda a pergunta do usuário abaixo, a qual está entre os caracteres aleatórios "N37463FYEIH64YBEJA~FLEWLK":
            N37463FYEIH64YBEJA~FLEWLK \n
            {question_user} \n
            N37463FYEIH64YBEJA~FLEWLK
        Responda a pergunta do usuário acima, a qual está entre os caracteres aleatórios "N37463FYEIH64YBEJA~FLEWLK".
        """),
        ('system', "Abaixo está um contexto recuperado por RAG a partir da conversa entre você, o agente, e o usuário. Esse conteúdo foi selecionado com base em similaridade com a pergunta atual. Use-o como referência para gerar uma resposta mais relevante e contextualizada. \n\n {history}"
        )
    ]

    guard_message = """
    Você deve analisar se o conteúdo da entrada do usuário está alinhado com o propósito do agente, que é exclusivamente conversar sobre vídeos de estudo presentes em uma playlist específica.

    O conteúdo **deve ser bloqueado** se:

    - Contém linguagem sexual, pornográfica ou de cunho impróprio.
    - Contém violência, discurso de ódio ou conteúdo sensível.
    - Está relacionado a programação, código, software, inteligência artificial, APIs ou ferramentas de desenvolvimento.
    - Foge do contexto de estudo ou da playlist fornecida.
    - Traz perguntas filosóficas, existenciais, religiosas, políticas ou pessoais que não estejam diretamente relacionadas ao conteúdo dos vídeos de estudo.
    - Usa palavras-chave como: "sexo", "nudez", "matar", "explodir", "script", "API", "GPT", "JavaScript", "código", "hacking", "hackear", "chatbot", "politica", "religião", "violência", "bomba", "terror", "dark web", entre outras relacionadas.
    - Menciona tópicos que não podem ser diretamente abordados com base no conteúdo da playlist de vídeos de estudo.

    Caso o conteúdo do usuário **não esteja claramente relacionado a vídeos de estudo da playlist**, ele também deve ser bloqueado.

    Se o conteúdo estiver de acordo com o objetivo (relacionado a vídeos de estudo da playlist), **não deve ser bloqueado**.

    No final, responda de forma objetiva apenas com:

    **O documento deve ser bloqueado: SIM**  
    ou  
    **O documento deve ser bloqueado: NAO**
    """

    template_blocked = """
    O conteúdo enviado pelo usuário foi classificado como inadequado ou fora do escopo educacional definido para este agente.

    Responda de forma cordial e respeitosa, explicando que você só pode interagir sobre conteúdos relacionados aos vídeos de estudo presentes na playlist fornecida.

    Informe que a mensagem recebida foge do escopo educacional e que, por esse motivo, não poderá ser respondida.

    Caso o usuário acredite que isso seja um engano, oriente-o a entrar em contato com o suporte responsável pela aplicação.

    Não forneça nenhum conteúdo, resposta ou sugestão que não esteja diretamente relacionada ao objetivo do agente.

    Seja claro, mas gentil.
    """

    main_prompt = ChatPromptTemplate.from_messages(messages)
    guard_prompt = ChatPromptTemplate.from_messages([
        ('system', guard_message),
        ('user', "Mensagem do usuário: \n {question_user}")])
    prompt_blocked = ChatPromptTemplate.from_template(template=template_blocked)
    structured_llm_guardrail = model.with_structured_output(GuardRail)


    def choose_chain(x: dict):

        binary_score = x["binary_score"].value
        if binary_score == "SIM":
            return (lambda x: x) | chain_blocked
        elif binary_score == "NAO":
            return (lambda x: x) | main_chain
    
    initial_chain = (
        {
        "question_user" : itemgetter("question_user"),
        "context" : itemgetter("context"),
        "binary_score" : itemgetter("question_user") | guard_prompt | structured_llm_guardrail | (lambda x: x.binary_score),
        "history": itemgetter("history")
        } 
        | RunnablePassthrough()
    )
    chain_blocked = prompt_blocked | model | StrOutputParser()
    main_chain = main_prompt | model | StrOutputParser()
    final_chain = initial_chain | RunnableLambda(choose_chain) | StrOutputParser()

    informations = {"question_user": question_user, "context": string_result_pc, "history": string_result_chroma}
    result_assistant = final_chain.invoke(informations)
    
    msg_user = Document(
        page_content=question_user,
        metadata={"author": "user"}
    )
    msg_assistant = Document(
        page_content=result_assistant,
        metadata={"author": "assistant"}
    )

    vector_store_chroma.add_documents([msg_user, msg_assistant])

    return result_assistant