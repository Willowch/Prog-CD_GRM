from __future__ import annotations

from pathlib import Path
import warnings

from recbole.quick_start import run_recbole

from CD_GRM.utils import (
    build_common_parser,
    load_raw_config,
    resolve_config_path,
    resolve_data_path,
    resolve_device,
    set_global_seed,
    slugify_dataset_name,
)

warnings.filterwarnings(
    "ignore",
    message="A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.*",
    category=FutureWarning,
)

BASE_DIR = Path(__file__).resolve().parent


def resolve_sasrec_config_path(config_arg: str) -> Path:
    """
    解析 SASRec 的配置文件路径。

    支持两种传法：
    1. 直接传一个真实存在的 yaml 路径
    2. 传数据集别名，如 all_beauty / sports_and_outdoors / toys_and_games
    """
    candidate = Path(config_arg)
    if candidate.exists():
        return candidate.resolve()

    config_map = {
        "all_beauty": Path("CD_GRM/config/sasrec_all_beauty.yaml"),
        "sports_and_outdoors": Path("CD_GRM/config/sasrec_sports_and_outdoors.yaml"),
        "toys_and_games": Path("CD_GRM/config/sasrec_toys_and_games.yaml"),
    }

    key = config_arg.strip().lower()
    config_path = config_map.get(key)
    if config_path is not None:
        if not config_path.exists():
            raise FileNotFoundError(f"SASRec config file not found: {config_path}")
        return config_path.resolve()

    # 如果不是 SASRec 专属别名，就退回你项目已有的通用解析逻辑
    return resolve_config_path(config_arg)


def build_sasrec_overrides(raw_config: dict) -> dict:
    """
    构造传给 RecBole SASRec 的覆盖配置。

    设计原则：
    - 数据路径、device、seed 与当前项目保持一致
    - 指标、topk、valid_metric 尽量与主模型保持一致
    - SASRec 参数使用一个“公开文档支持 + 对你项目相对公平”的起始配置
    """
    dataset_slug = slugify_dataset_name(raw_config["dataset"])
    ckpt_dir = BASE_DIR / "checkpoints" / "sasrec" / dataset_slug

    # 为了和你的主模型更公平，这里优先取 yaml 里手动写的值；
    # 如果没写，再给一个较稳的起始值。
    hidden_size = raw_config.get("hidden_size", 128)
    inner_size = raw_config.get("inner_size", hidden_size * 4)

    return {
        # -------------------------
        # 基础运行信息
        # -------------------------
        "data_path": raw_config["data_path"],
        "device": raw_config["device"],
        "seed": raw_config["seed"],
        "reproducibility": raw_config.get("reproducibility", True),

        # -------------------------
        # 训练设置
        # -------------------------
        "epochs": raw_config.get("epochs", 100),
        "train_batch_size": raw_config.get("train_batch_size", 256),
        "eval_batch_size": raw_config.get("eval_batch_size", 256),
        "learning_rate": raw_config.get("learning_rate", 1e-3),
        "weight_decay": raw_config.get("weight_decay", 0.0),
        "checkpoint_dir": str(ckpt_dir),
        "show_progress": raw_config.get("show_progress", True),

        # -------------------------
        # 评测设置：尽量和主模型统一
        # -------------------------
        "topk": raw_config.get("topk", [5, 10, 20]),
        "metrics": raw_config.get("metrics", ["Recall", "NDCG"]),
        "valid_metric": raw_config.get("valid_metric", "NDCG@10"),
        "eval_step": raw_config.get("eval_step", 1),
        "stopping_step": raw_config.get("stopping_step", 10),

        # -------------------------
        # SASRec 参数
        # -------------------------
        "hidden_size": hidden_size,
        "inner_size": inner_size,
        "n_layers": raw_config.get("n_layers", 2),
        "n_heads": raw_config.get("n_heads", 4),
        "hidden_dropout_prob": raw_config.get("hidden_dropout_prob", 0.2),
        "attn_dropout_prob": raw_config.get("attn_dropout_prob", 0.2),
        "hidden_act": raw_config.get("hidden_act", "gelu"),
        "layer_norm_eps": raw_config.get("layer_norm_eps", 1e-12),
        "initializer_range": raw_config.get("initializer_range", 0.02),
        "loss_type": raw_config.get("loss_type", "CE"),

        # -------------------------
        # CE 训练下不需要负采样
        # -------------------------
        "train_neg_sample_args": None,
    }


def main() -> None:
    args = build_common_parser("Train SASRec.").parse_args()

    config_path = resolve_sasrec_config_path(args.config)
    raw_config = load_raw_config(config_path)

    data_file = resolve_data_path(raw_config)
    device = resolve_device(raw_config, args.device)
    set_global_seed(raw_config["seed"])

    sasrec_overrides = build_sasrec_overrides(raw_config)

    print("=" * 60)
    print(f"Training SASRec on {raw_config['dataset']}")
    print(f"Config    : {config_path}")
    print(f"Data file : {data_file}")
    print(f"Device    : {device}")
    print(f"Save dir  : {sasrec_overrides['checkpoint_dir']}")
    print("=" * 60)

    run_recbole(
        model="SASRec",
        dataset=raw_config["dataset"],
        config_file_list=[str(config_path)],
        config_dict=sasrec_overrides,
        saved=True,
    )


if __name__ == "__main__":
    main()