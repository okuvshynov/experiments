import logging
import yaml

def open_yaml(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
        logging.info(f'Config: {config}')
        return config
