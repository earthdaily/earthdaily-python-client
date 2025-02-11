import logging


class LoggerConfig:
    """
    Logger configuration class for setting up a basic console logger.

    Attributes:
        logger_name (str): Name of the logger instance.
        log_level (int): Logging level for the logger instance.
    """

    def __init__(self, logger_name: str = "earthdatastore", log_level: int = logging.INFO):
        """
        Initializes LoggerConfig with a specified logger name and level.

        Args:
            logger_name (str): Name of the logger instance. Default is 'earthdatastore'.
            log_level (int): Logging level. Default is logging.INFO.
        """
        self.logger_name = logger_name
        self.log_level = log_level
        self.logger = self._initialize_logger()

    def _initialize_logger(self) -> logging.Logger:
        """
        Initializes and configures a console logger.

        Returns:
            logging.Logger: Configured logger instance.
        """
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(self.log_level)

        if not logger.hasHandlers():
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.log_level)
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            console_handler.setFormatter(formatter)

            logger.addHandler(console_handler)

        return logger

    def get_logger(self) -> logging.Logger:
        """
        Provides the configured logger instance.

        Returns:
            logging.Logger: The logger instance.
        """
        return self.logger
