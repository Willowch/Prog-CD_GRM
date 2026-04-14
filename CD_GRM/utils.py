from __future__ import annotations
import os
import random
import numpy as np
import yaml
import re
import argparse
from pathlib import Path
import torch


BASE_DIR = Path(__file__).resolve()  #当前脚本所在目录的绝对路径

#设置全局随机数
def set_global_seed(seed: int = 42) -> None:
    """
    全局固定所有相关的随机数种子，确保扩散模型与图游走过程 100% 可复现。
    
    Args:
        seed (int): 随机数种子值，默认为 42
    """
    # 1. 固定 Python 内置随机库种子 (影响基础数据结构如 list 的 random 操作)
    random.seed(seed)
    
    # 2. 固定 NumPy 随机种子 (影响数据预处理、图游走采样等操作)
    np.random.seed(seed)
    
    # 3. 固定 PyTorch CPU 端的随机种子 (影响 CPU 上的 Tensor 初始化操作)
    torch.manual_seed(seed)
    
    # 4. 固定 PyTorch 当前 GPU 的随机种子
    torch.cuda.manual_seed(seed)
    
    # 5. 固定 PyTorch 所有可用 GPU 的随机种子 (兼容未来可能的多卡扩展)
    torch.cuda.manual_seed_all(seed)
    
    # 6. 配置 CuDNN 后端，确保 GPU 卷积与线性层计算结果的确定性
    # 注意: deterministic=True 可能会带来极少量的性能损耗，但在研究复现阶段是必须的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 7. 固定 Python Hash 种子，保证字典遍历顺序等一致性
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 8. (可选但推荐) 配置 PyTorch 强制使用确定性算法，防止某些具有不确定性原子操作的算子引发误差
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
    # torch.use_deterministic_algorithms(True)
    
    print(f"[Project Core] Global random seed dynamically fixed to: {seed}")
#解析参数
def build_common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Dataset alias (e.g. all_beauty).",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override, e.g. cpu / cuda / cuda:0.",
    )

    return parser


def parse_train_args() -> argparse.Namespace:
    parser = build_common_parser("Train CD_GRM with Optuna.")

    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of Optuna trials.",
    )

    return parser.parse_args()


def parse_test_args() -> argparse.Namespace:
    parser = build_common_parser("Evaluate CD_GRM on the test split.")
    parser.add_argument(
        "--trial",
        type=int,
        default=None,
        help="Optional Optuna trial number. Used when --checkpoint is not provided.",
    )
    return parser.parse_args()


#根据命令行传入的数据集名，返回对应配置文件路径
def resolve_config_path(config_arg: str) -> Path:
    config_map = {
        "all_beauty": Path("CD_GRM/config/cd_grm_all_beauty.yaml"),
        "sports_and_outdoors": Path("CD_GRM/config/cd_grm_sports_and_outdoors.yaml"),
        "toys_and_games": Path("CD_GRM/config/cd_grm_toys_and_games.yaml"),
    }

    key = config_arg.strip().lower()# 去掉首尾空格，并转成小写，避免大小写影响匹配

    if key not in config_map:
        raise FileNotFoundError(
            f"Unsupported dataset name: {config_arg}for config file. "
            f"Expected one of: all_beauty, sports_and_outdoors, toys_and_games"
        )
    config_path = config_map[key]
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    return config_path.resolve()#返回配置文件的绝对规范路径
#读取 YAML 配置文件并返回字典
#解析.inter目录，并写回cofig
def resolve_data_path(raw_config: dict) -> Path:
    dataset_name = raw_config["dataset"]  # 从原始配置中读取数据集名称
    data_map = {
        "all_beauty": Path("CD_GRM/data/All_Beauty/All_Beauty.inter"),
        "sports_and_outdoors": Path("CD_GRM/data/Sports_and_Outdoors/Sports_and_Outdoors.inter"),
        "toys_and_games": Path("CD_GRM/data/Toys_and_Games/Toys_and_Games.inter"),
    }
    key = dataset_name.strip().lower()  # 去掉首尾空格，并转成小写，避免大小写影响匹配
    if key not in data_map:
        raise FileNotFoundError(
            f"Unsupported dataset name: {dataset_name}for data file. "
            f"Expected one of: all_beauty, sports_and_outdoors, toys_and_games"
        )
    data_path = data_map[key]
    if not data_path.exists():
        raise FileNotFoundError(f"data file not found: {data_path}")
    raw_config["data_path"] = str(data_path)
    return data_path.resolve()  # 返回配置文件的绝对规范路径
#解析device；写回config并返回
def resolve_device(raw_config: dict, device_override: str | None) -> torch.device:
    requested = device_override or raw_config.get("device", "cpu")  # 优先使用命令行传入的 device；否则使用配置文件中的 device；再否则默认 cpu

    if requested.startswith("cuda") and not torch.cuda.is_available():  # 如果用户请求的是 CUDA，但当前环境没有可用 GPU
        print("[Info] CUDA is unavailable, falling back to cpu.")  # 打印提示信息
        requested = "cpu"  # 自动回退到 cpu

    raw_config["device"] = requested  # 把最终决定的 device 写回配置字典
    return torch.device(requested)  # 返回 PyTorch 的 device 对象
def load_raw_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
#数据名清洗
def slugify_dataset_name(dataset_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", dataset_name.lower())
    # 先把数据集名转成小写
    # 再用正则把所有“不是小写字母和数字”的连续字符替换成下划线 _
    # 例如 "All-Beauty V2" -> "all_beauty_v2"
    return slug.strip("_")# 去掉字符串首尾多余的下划线后返回
