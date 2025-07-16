import os
import json
import logging
import time
from tavily import TavilyClient, MissingAPIKeyError, InvalidAPIKeyError, UsageLimitExceededError
from httpx import TimeoutException, HTTPError

logger = logging.getLogger(__name__)
_tavily_client = None
_SEARCH_CONFIG = None

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'chatbot', 'rag', 'config', 'config.json')

# Utilidad para limpiar la consulta
def clean_query(query: str) -> str:
    """
    Limpia y valida la consulta según las mejores prácticas de Tavily.
    """
    cleaned = query.strip().replace('\n', ' ')
    
    # Validar longitud según mejores prácticas de Tavily (máximo 400 caracteres)
    if len(cleaned) > 400:
        logger.warning(f"Consulta muy larga ({len(cleaned)} caracteres), truncando a 400")
        cleaned = cleaned[:400]
    
    return cleaned

def get_tavily_client():
    global _tavily_client
    if _tavily_client is None:
        try:
            _tavily_client = TavilyClient()
            logger.info("Cliente Tavily inicializado correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar cliente Tavily: {e}")
            raise
    return _tavily_client

def get_search_config():
    global _SEARCH_CONFIG
    if _SEARCH_CONFIG is None:
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                config = json.load(f)
                websearch_config = config.get('websearch', {})
                _SEARCH_CONFIG = {
                    'include_domains': websearch_config.get('include_domains', []),
                    'country': websearch_config.get('country', 'colombia'),
                    'max_results': websearch_config.get('max_results', 3),
                    'chunks_per_source': websearch_config.get('chunks_per_source', 3),
                    'search_depth': websearch_config.get('search_depth', 'advanced')
                }
                logger.info(f"Configuración de búsqueda cargada: {_SEARCH_CONFIG}")
        except Exception as e:
            logger.warning(f"No se pudo leer la configuración de búsqueda web: {e}")
            _SEARCH_CONFIG = {
                'include_domains': [],
                'country': 'colombia',
                'max_results': 3,
                'chunks_per_source': 3,
                'search_depth': 'advanced'
            }
    return _SEARCH_CONFIG

def search_web(query: str) -> list:
    """
    Realiza búsqueda web usando las mejores prácticas de Tavily.
    
    Args:
        query (str): La consulta de búsqueda
    
    Returns:
        list: Lista de resultados de búsqueda ordenados por relevancia
    
    Raises:
        Exception: Para errores críticos que requieren intervención
    """
    query = clean_query(query)
    
    if not query:
        logger.error("Consulta vacía después de limpieza")
        return []
    
    try:
        client = get_tavily_client()
        search_config = get_search_config()
    except Exception as e:
        logger.error(f"Error en inicialización: {e}")
        return []
    
    max_retries = 3
    base_wait_time = 1
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"Intento {attempt + 1} de búsqueda para: '{query[:50]}...'")
            
            response = client.search(
                query=query,
                search_depth=search_config['search_depth'],
                max_results=search_config['max_results'],
                chunks_per_source=search_config['chunks_per_source'],
                include_raw_content=True,  # Mejores prácticas con search_depth=advanced
                include_domains=search_config['include_domains'],
                country=search_config['country'],
            )
            
            results = response.get('results', [])
            response_time = response.get('response_time', 'N/A')
            
            logger.info(f"Búsqueda exitosa: {len(results)} resultados en {response_time}s para '{query[:30]}...'")
            
            # Validar calidad de resultados
            valid_results = []
            for result in results:
                if result.get('content') or result.get('raw_content'):
                    valid_results.append(result)
                else:
                    logger.debug(f"Resultado sin contenido ignorado: {result.get('url', 'URL desconocida')}")
            
            logger.info(f"Resultados válidos: {len(valid_results)}/{len(results)}")
            return valid_results
            
        except (MissingAPIKeyError, InvalidAPIKeyError) as e:
            logger.error(f"Error de autenticación con Tavily API: {e}")
            logger.error("Verifique que TAVILY_API_KEY esté configurada correctamente")
            return []  # No reintentar errores de API key
            
        except UsageLimitExceededError as e:
            logger.error(f"Límite de uso de Tavily API excedido: {e}")
            logger.error("Verifique su plan y límites de API en https://app.tavily.com")
            return []  # No reintentar límites excedidos
            
        except (TimeoutException, HTTPError) as e:
            if attempt == max_retries - 1:
                logger.error(f"Error de red persistente después de {max_retries} intentos: {e}")
                return []
            else:
                # Backoff exponencial con jitter
                wait_time = base_wait_time * (2 ** attempt) + (attempt * 0.1)
                logger.warning(f"Error de red en intento {attempt + 1}, reintentando en {wait_time:.1f}s: {e}")
                time.sleep(wait_time)
                
        except ValueError as e:
            if "Query is too long" in str(e):
                logger.error(f"Consulta demasiado larga: {e}")
                return []
            else:
                logger.error(f"Error de valor en búsqueda: {e}")
                return []
                
        except Exception as e:
            logger.error(f"Error inesperado en búsqueda web (intento {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                logger.error("Se agotaron todos los reintentos")
                return []
            else:
                wait_time = base_wait_time * (attempt + 1)
                logger.warning(f"Reintentando en {wait_time}s...")
                time.sleep(wait_time)