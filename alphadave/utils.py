import logging
import os
from logging import FileHandler, Logger, StreamHandler

import torch


def get_logger(save_dir: str) -> Logger:
    os.makedirs(save_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        format='[%(asctime)s] %(message)s',
        handlers=[StreamHandler(), FileHandler(os.path.join(save_dir, 'log.txt'))],
    )
    return logging.getLogger()


def save_checkpoint(ckpt_to_save: dict, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    tag = ckpt_to_save['episode']
    torch.save(ckpt_to_save, os.path.join(save_dir, f'alphadave-{tag}.pt'))


def load_checkpoint(ckpt_to_load: str) -> dict:
    return torch.load(ckpt_to_load)
