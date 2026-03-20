"""
示例脚本模块
"""

from .train import main as train_main
from .evaluate import main as evaluate_main
from .visualize import main as visualize_main

__all__ = ['train_main', 'evaluate_main', 'visualize_main']