import yaml
import os
import argparse # Keep argparse for Namespace

class Config:
    def __init__(self, config_name="mnist"):
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.config_name = config_name
        self.args = argparse.Namespace() # This will hold the combined config and CLI args
        self.load_config()

    def load_config(self):
        config_path = os.path.join(os.path.dirname(self.base_path), "configs", f"{self.config_name.lower()}.yaml")

        if not os.path.exists(config_path):
            raise ValueError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            yaml_config = yaml.safe_load(f)

        # Populate self.args with values from the YAML config
        for key, value in yaml_config.items():
            setattr(self.args, key, value)

    def __getattr__(self, name):
        if hasattr(self.args, name):
            return getattr(self.args, name)
        raise AttributeError(f"'Config' object has no attribute '{name}'")