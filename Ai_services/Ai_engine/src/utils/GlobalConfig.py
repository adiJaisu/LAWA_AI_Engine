"""
GlobalConfig module provides a singleton-like class to manage global configuration settings
for the application. It allows setting and retrieving configuration values using a dictionary
internally. The configuration is expected to be provided as a list of metadata dictionaries,
each containing parameter names and their corresponding values, as defined by the Constants class.

Classes:
    GlobalConfig: Handles storage and retrieval of global configuration parameters.

Usage:
    config = GlobalConfig()
    config.set_value(config_metadata)
    value = config.get('parameter_name', default_value)
Author: HCLTech
"""

from src.constant.constants import Constants
class GlobalConfig:
    """
    Global configuration class for the application.
    This class holds configuration settings that can be accessed globally.
    """

    def __init__(self):
        self._config = {}

    def set_value(self,config_metadata):
        """Set a configuration value."""
        self._config = {item[Constants.PARAMETER]: item[Constants.VALUE] for item in config_metadata[Constants.CONFIG_METADATA]}


    def get(self,key, default=None):
        """Get a configuration value, returning default if not set."""
        return self._config.get(key, default)