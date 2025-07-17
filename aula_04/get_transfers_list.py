from operator import itemgetter
import os
from typing import List
from enum import Enum

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from llama_parse import LlamaParse
from pydantic import Field, BaseModel

load_dotenv(override=True)


class CategoriaGasto(Enum):
    MORADIA         = "Moradia"
    ALIMENTACAO     = "Alimentação"
    TRANSPORTE      = "Transporte"
    SAUDE           = "Saúde"
    EDUCACAO        = "Educação"
    LAZER           = "Lazer e Entretenimento"
    IMPOSTOS        = "Impostos e Obrigações Legais"
    PESSOA_FISICA   = "Transação com pessoa física"
    OUTROS          = "Outros"


class OrigemTransacao(Enum):
    PIX                     = "PIX"
    TRANSFERENCIA           = "Transferência"
    DEPOSITO                = "Depósito"
    SAQUE                   = "Saque em dinheiro"
    COMPRA_CARTAO           = "Compra com cartão"
    PAGAMENTO_BOLETO        = "Pagamento de boleto"
    ESTORNO                 = "Estorno"
    OUTROS                  = "Outros"


class Transferencia(BaseModel):
    valor: float = Field(..., description="O valor da transferência, o qual é um inteiro (negativo, positivo ou nulo).")
    origem: OrigemTransacao = Field(..., description="Forma como a transação foi realizada, como PIX, transferência, compra com cartão, etc.")
    categoria: CategoriaGasto = Field(..., description="Categoria da pessoa ou entidade que enviou ou recebeu a transação.")


class Extrato(BaseModel):
    extrato: List[Transferencia] = Field(..., description="Lista completa das transferências realizadas e recebidas no extrato bancário.")


def get_extrato(path: str) -> List[Transferencia]:

    parser = LlamaParse(
        api_key = os.getenv("LLAMA_CLOUD_API_KEY"),
        verbose=True,
        premium_mode=False,
        fast_mode=True
    )

    result = parser.parse(file_path=path).get_text_documents(split_by_page=False)
    extrato = str()
    for document in result:
        extrato += document.text

    model = ChatOpenAI(model="gpt-4o")
    structured_model = model.with_structured_output(Extrato)

    message = """
    Você é um parser de extrato bancário genérico: recebe um bloco de texto (várias linhas) de qualquer banco e deve devolver **somente** o código Python que constrói:

        Extrato(extrato=[Transferencia(...), Transferencia(...), ...])

    Sem nenhuma outra saída (nem JSON, nem texto explicativo).

    **Como fazer:**

    1. **Detectar linhas de transação**  
    - Cada linha relevante referente à transação costuma ter: data (DD/MM/AAAA), descrição (nome do estabelecimento ou pessoa) e valor (formato brasileiro, ex. 1.234,56 ou -123,45).  
    - Cada transação pode ter mais de uma linha correspondente.

    2. **Manter o sinal correto**  
    - Se o valor vier com “-”, use valor negativo; caso contrário, positivo.  

    3. **Mapear `origem: OrigemTransacao`** (busca case-insensitive na descrição). Exemplos:
    - Contém “pix” → `PIX`  
    - Contém “estorno” → `ESTORNO`  

    4. **Mapear `categoria: CategoriaGasto`** (baseado na descrição do terceiro):  
    - Palavras-chave de MORADIA: “aluguel”, “condomínio”, “imobiliária”  
    - ALIMENTACAO: “supermercado”, “mercado”, “restaurante”, “ifood”, "açaí”
    - TRANSPORTE: “uber”, “99”, “gasolina”, “posto”, “ônibus”, “metro”  
    - SAUDE: “farmácia”, “drogaria”, “hospital”, “clínica”, “laboratório”  
    - EDUCACAO: “escola”, “faculdade”, “curso”, “colegial”  
    - LAZER: “cinema”, “streaming”, “show”, “bar”  
    - IMPOSTOS: “imposto”, “ir”, “taxa”
    - PESSOA_FISICA: transferência para CPF ou nome próprio de pessoa  
    - OUTROS: caso não se enquadre em nenhum acima
    """


    prompt = ChatPromptTemplate.from_messages(
        [('system'), (message),
        ('user', "{extrato}")]
    )

    chain = (
        {"extrato": itemgetter("extrato")}
        | prompt
        | structured_model
        | (lambda x: x.extrato)
    )

    resposta = chain.invoke({"extrato": extrato})

    return resposta