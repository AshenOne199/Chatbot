# ./chatbot/rag/QA_Cohere_Handler.py

import os
import re
import random
import logging
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_cohere.chat_models import ChatCohere
from chatbot.rag.utils.singleton_meta import SingletonMeta
from chatbot.rag.handlers.base_handler import BaseQAHandler
from chatbot.rag.utils import utils
from chatbot.rag.utils.patterns import (
    prompt_template,
    greetings,
    greeting_messages,
    farewell,
    farewell_messages,
    gratefulness,
    gratefulness_messages,
)
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.pydantic_v1 import BaseModel, Field

load_dotenv()

# Configuración de la API Key de Tavily
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")  # Asegúrate de tener TAVILY_API_KEY en tu .env

# Instancia de búsqueda en internet
internet_search = TavilySearchResults()
internet_search.name = "internet_search"
internet_search.description = "Returns a list of relevant document snippets for a textual query retrieved from the internet."
internet_search.include_domains = ["https://www.udistrital.edu.co"]
internet_search.search_depth = "advanced"


class TavilySearchInput(BaseModel):
    query: str = Field(description="Retorna unicamente la respuesta a lo que te preguntan, si no se encuentra la respuesta sugiere al usuario que busque en el dominio de https://www.udistrital.edu.co")
internet_search.args_schema = TavilySearchInput

logger = logging.getLogger(__name__)

class QA_CohereHandler(BaseQAHandler, metaclass=SingletonMeta):
    """
    Handler to manage interactions with the Cohere model for generating responses
    based on documents retrieved using TF-IDF.
    """
    
    def __init__(self, model: str, temperature: float, max_tokens: int, docs_directory: str,
                 chunk_size: int = 500, chunk_overlap: int = 0):
        """
        Initializes the handler with model parameters, prompt template, and document database.

        Args:
            model (str): The type of Cohere model.
            temperature (float): Level of randomness for response generation.
            max_tokens (int): Maximum number of tokens in the generated response.
            docs_directory (str): Directory path to the documents database.
            chunk_size (int): Size of the chunks when splitting documents for retrieval.
            chunk_overlap (int): Size of the overlap between document chunks.
        """
        # Avoid multiple initializations
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True

        # Model parameter configuration
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info(f'Temperature: {temperature}')
        logger.info(f'Max Tokens: {max_tokens}')

        # Load prompt template, documents, LLM, and QA retriever
        self.load_prompt_template()
        self.tfidf_retriever = utils.load_documents_database(docs_directory, chunk_size, chunk_overlap)
        self.load_llm()
        self.load_qa()
        
        logger.info('Cohere Handler creado correctamente.')
        
    def load_prompt_template(self):
        """
        Loads the prompt template for generating queries.
        """
        try:
            self.prompt = PromptTemplate(
                template = prompt_template,
                input_variables = ["context", "question"]
            )
            logger.info('PromptTemplate cargado correctamente.')
        except Exception as e:
            logger.error('Ha ocurrido un error al cargar el PromptTemplate.', exc_info=True)

    def load_llm(self):
        """
        Loads the Cohere LLM model to generate responses.
        """
        try:
            print(self.max_tokens)
            self.llm = ChatCohere(
                model=self.model,
                temperature=self.temperature,
                cohere_api_key=os.getenv('COHERE_API_KEY'),
                max_tokens=self.max_tokens
            )
            logger.info('LLM cargado correctamente.')
        except Exception as e:
            logger.error(f'Ha ocurrido un error al cargar el LLM {self.model}.', exc_info=True)
        
    def load_qa(self):
        """
        Initializes the RetrievalQA chain for querying the document database.
        """
        try:
            self.qa = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.tfidf_retriever,
                verbose=False,
                chain_type_kwargs={
                    "verbose": False,
                    "prompt": self.prompt
                }
            )
            logger.info('RetrievalQA cargado correctamente.')
        except Exception as e:
            logger.error('Ha ocurrido un error al cargar el RetrievalQA.', exc_info=True)

    def get_answer(self, query: str) -> str:
        """
        Genera una respuesta para la consulta dada usando el modelo Cohere o búsqueda en internet si se solicita.

        Args:
            query (str): Consulta o pregunta del usuario.

        Returns:
            str: Respuesta generada por el modelo Cohere o por Tavily.
        """
        try:
            if any(re.match(pattern, query.lower()) for pattern in greetings):
                response =  random.choice(greeting_messages)
            elif any(re.match(pattern, query.lower()) for pattern in farewell):
                response = random.choice(farewell_messages)
            elif any(re.match(pattern, query.lower()) for pattern in gratefulness):
                response = random.choice(gratefulness_messages)
            elif query.lower().startswith("internet:"):
                # Búsqueda en la web usando Tavily
                consulta = query[len("internet:"):].strip()
                resultados = internet_search.invoke({"query": consulta})
                # Puedes personalizar el formato de la respuesta según lo que devuelva Tavily
                if resultados and isinstance(resultados, list):
                    response = "\n\n".join([r.get("content", str(r)) for r in resultados])
                else:
                    response = str(resultados)
            else:
                response = self.qa.run({'query': query})
            return response
        except Exception as e:
            logger.error('Ha ocurrido un error en la ejecución del Query.', exc_info=True)
            return "Lo siento, ha ocurrido un error al procesar tu consulta."
