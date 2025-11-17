"""
File: utils.py
Chứa các hàm tiện ích: set_global_seed, setup_logging.
"""
import os
import random
import numpy as np
import logging
import sys
import optuna

def setup_logging():
    """
    Thiết lập cấu hình logging cơ bản.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Logging configured.")
    
    # Tắt thông báo rác của Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)


def set_global_seed(seed: int = 42):
    """
    Thiết lập seed cho các thư viện để đảm bảo tái lập kết quả.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.info(f"Global seed set to {seed}.")