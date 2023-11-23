import logging

def setup_logging(log_file='audio_stream.log', level=logging.DEBUG):
    """Setup logging configuration
    Arfs:
        log_file: name of the log file
        level: logging level
        Returns:
            logger: logging.Logger object
    """
    
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()

    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger