import json

class ConfigManager:
    _instance = None

    def __new__(cls, config_file):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            with open(config_file, "r") as f:
                cls._instance.config = json.load(f)
        return cls._instance

    def get_config(self):
        return self.config



