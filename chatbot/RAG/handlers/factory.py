# ./chatbot/rag/handler_factory.py

import logging
from chatbot.rag.handlers import aws_bedrock_handler, cohere_handler

logger = logging.getLogger(__name__)

def get_qa_handler(bot_type: str, bot_config: dict):
    """
    Returns the appropriate QA handler based on the bot type.

    Args:
        bot_type (str): The type of bot ('aws_bedrock' or 'cohere').
        bot_config (dict): Configuration parameters for initializing the bot handler.

    Returns:
        QA_AWS_Bedrock_Handler or QA_Cohere_Handler: An instance of the specified bot handler.

    Raises:
        ValueError: If the bot type is not supported.
    """
    
    if bot_type == 'aws_bedrock':
        logger.info('Inicializando AWS Bedrock Handler...')
        return aws_bedrock_handler.QA_AwsBedrockHandler(**bot_config)
    elif bot_type == 'cohere':
        logger.info('Inicializando Cohere Handler (solo búsqueda web)...')
        # Cohere handler ahora solo necesita model, temperature y max_tokens
        cohere_config = {
            'model': bot_config['model'],
            'temperature': bot_config['temperature'],
            'max_tokens': bot_config['max_tokens']
        }
        return cohere_handler.QA_CohereHandler(**cohere_config)
    else:
        logger.error('No se ha podido inicializar ningun Handler.')
        raise ValueError(f"Unsupported bot type: {bot_type}")
