import logging
import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

class LogTool:
    _logger = None

    @staticmethod
    def _get_logger():
        if LogTool._logger is None:
            # 确保日志目录存在
            logDir = "app/logs"
            if not os.path.exists(logDir):
                os.makedirs(logDir)
            
            LogTool._logger = logging.getLogger("ASRBrain")
            LogTool._logger.setLevel(logging.INFO)

            # 格式化
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

            # app.log (全量日志，按天切分)
            appHandler = TimedRotatingFileHandler(
                os.path.join(logDir, "app.log"),
                when="midnight",
                interval=1,
                backupCount=30,
                encoding="utf-8"
            )
            appHandler.setFormatter(formatter)
            LogTool._logger.addHandler(appHandler)

            # error.log (独立错误日志)
            errHandler = TimedRotatingFileHandler(
                os.path.join(logDir, "error.log"),
                when="midnight",
                interval=1,
                backupCount=30,
                encoding="utf-8"
            )
            errHandler.setLevel(logging.ERROR)
            errHandler.setFormatter(formatter)
            LogTool._logger.addHandler(errHandler)

            # 控制台输出 (开发调试用)
            consoleHandler = logging.StreamHandler()
            consoleHandler.setFormatter(formatter)
            LogTool._logger.addHandler(consoleHandler)

        return LogTool._logger

    @staticmethod
    def info(message):
        LogTool._get_logger().info(message)

    @staticmethod
    def error(message, exception=None):
        if exception:
            LogTool._get_logger().error(f"{message} | Exception: {str(exception)}", exc_info=True)
        else:
            LogTool._get_logger().error(message)

    @staticmethod
    def debug(message):
        LogTool._get_logger().debug(message)
