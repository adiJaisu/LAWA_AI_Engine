import configparser
import os
from src.constant.constants import Constants 

class ConfigManager:
    """
    Singleton class to manage configuration loading.
    """
    def __new__(self):
        if not hasattr(self, 'instance'):
            file_name = Constants.CONFIG_FILE_PATH
            self.parser = configparser.ConfigParser()
            self.parser.optionxform = str
            self.config = self.parser
            if os.path.exists(file_name):
                self.config.read(file_name, encoding=Constants.UTF_8_ENCODING)
            else:
                # print(f" [WARNING] Config file not found at {file_name}. Using defaults.")
                pass

            self.instance = super(ConfigManager, self).__new__(self)

        return self.config

class ReadConfigFile:
    def __init__(self):
        self.obj_config = ConfigManager()

    def get_env_config(self, param):
        # Prioritize environment variables from Docker/System
        env_val = os.environ.get(param)
        if env_val is not None:
            return env_val

        try:
            env_config = self.obj_config[Constants.DEFAULT_ENVIRONMENT]
            return env_config.get(param)
        except Exception as e:
            return None

    def get_value_config(self, section, param):
        try:
            env_config = self.obj_config[section]
            return env_config.get(param)
        except Exception as e:
            return None

cfg = ReadConfigFile()
