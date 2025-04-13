import logging
import logging.config


def setup_logger():
    # Define la configuración básica del logger
    logging.basicConfig(
        level=logging.INFO,  # Nivel de logs que quieres capturar (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Formato del log
        handlers=[
            # logging.FileHandler("app.log"),  # Guarda los logs en un archivo
            logging.StreamHandler(),  # También imprime los logs en la consola
        ],
    )

    # Retorna el logger principal
    logger = logging.getLogger("main_logger")
    return logger


# Llama esta función para inicializar el logger una vez al inicio
logger = setup_logger()
