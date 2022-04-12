"""a class that imports the python logger and creates logging"""
import logging
from logging.handlers import TimedRotatingFileHandler
import sys
from hydroDL.core.read import get_real_path
from hydroDL.core import APP_LOGGER_NAME


LOG_FILE_PATH = get_real_path("../logs/hydroDL.log")


class Logger(logging.Logger):
    def __init__(self, name, level=logging.NOTSET):
        """
        A child class inheriting Logger
        :param name: name of the logger
        :param level: level that you're logging
        """
        return super(Logger, self).__init__(name, level)

    def exception(self, msg, *args, **kwargs):
        """
        An overload of the exception logging class to allow the code to throw an exception
        :param msg: The message you want to throw
        :param args:
        :param kwargs:
        :return:
        """

        return super(Logger, self).exception(msg, *args, **kwargs)

    def close(self):
        """
        Closes the logger
        :return:
        """
        logging.shutdown()


def get_logger(module_name):
    """
    Gets the logger module
    @param module_name: the name of the log
    returns: the logger for this module
    """
    return logging.getLogger(APP_LOGGER_NAME).getChild(module_name)


def setup_app_logger(logger_name=APP_LOGGER_NAME, file_name=LOG_FILE_PATH):
    """
    Sets up the app logger
    @param logger_name: the name of the log
    @param file_name: the path to the log
    returns: the implemented log
    """
    logging.setLoggerClass(Logger)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    info_formatter = logging.Formatter("%(name)-24s: %(levelname)-8s %(message)s")
    debug_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    """Configuring the streamHandler for console output"""
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(info_formatter)
    logger.addHandler(sh)

    """Configuring the rotating file handler for debug statements"""
    h = TimedRotatingFileHandler(file_name, when="midnight", interval=1, backupCount=10)
    h.setLevel(logging.DEBUG)
    h.suffix = "%Y-%m-%d"
    h.setFormatter(debug_formatter)
    logger.addHandler(h)
    return logger
