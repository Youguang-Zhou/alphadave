import json
import os

import psutil


def get_pid(process_name: str) -> int:
    for proc in psutil.process_iter():
        if proc.name() == process_name:
            return proc.pid
    raise RuntimeError(f'Process {process_name} not open!')


def get_plant_vocab() -> dict:
    with open(os.path.join('pvzgym', 'vocab.json'), encoding='utf-8') as f:
        return json.load(f)
