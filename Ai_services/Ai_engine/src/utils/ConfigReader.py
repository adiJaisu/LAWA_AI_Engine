"""
Manages application configuration by reading settings from a `.ini` file.

This module provides a **singleton `ConfigManager`** to ensure the configuration
is loaded only once and is accessible globally. It offers robust methods to:
- **Load configuration**: Reads parameters from the specified `config.ini` file.
- **Retrieve environment settings**: Access values from the `[DEFAULT]` section.
- **Fetch specific parameters**: Get values from any designated configuration section.
Author: HCLTech
"""

import configparser
import os
from src.constant.constants import Constants 
from src.Exception.Exception import FileNotFoundException


class ConfigManager:
    """
    This class opens the configuration file and
    reads all the configuration provided inside it
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
                raise FileNotFoundException('No Config file found!')

            self.instance = super(ConfigManager, self).__new__(self)

        return self.config


class ReadConfigFile:
    """
    This class reads specific configurations based on user requirement
    """

    objConfig = ''

    def __init__(self):
        self.obj_config = ConfigManager()

    def get_env_config(self, param):
        """

        Arguments:
            param -- Parameter to retrieve from Default Env

        Returns:
            Parameter value
        """
        try:
            env_config = self.obj_config[Constants.DEFAULT_ENVIRONMENT]
            return env_config[param]
        except NotImplementedError as e:
            raise Exception("The selected parameter is not correct")

    def get_value_config(self, section, param):
        """

        Arguments:
            section -- Environment to retrieve value from
            param -- Parameter to retrieve from Env

        Returns:
            Parameter value
        """
        try:
            env_config = self.obj_config[section]
            return env_config[param]
        except NotImplementedError as e:
            raise Exception("The selected environment is not implemented")


# Create an instance of ReadConfigFile to be used throughout the program
cfg = ReadConfigFile()