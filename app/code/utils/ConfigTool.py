import yaml
import os
from utils.LogTool import LogTool

class ConfigTool:
    _config = {}

    @staticmethod
    def load(fileName):
        """
        加载配置文件，默认从 app/config/ 目录下读取
        """
        try:
            configPath = os.path.join("app/config", fileName)
            if not os.path.exists(configPath):
                LogTool.error(f"Config file not found: {configPath}")
                return False
            
            with open(configPath, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                if data:
                    ConfigTool._config.update(data)
                    LogTool.info(f"Successfully loaded config: {fileName}")
                    return True
        except Exception as e:
            LogTool.error(f"Failed to load config: {fileName}", e)
            return False

    @staticmethod
    def get(key, defaultValue=None):
        """
        通过点分隔符获取配置，例如 ConfigTool.get("modelConfig.modelSize")
        """
        keys = key.split('.')
        value = ConfigTool._config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return defaultValue
