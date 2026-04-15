from __future__ import annotations

import argparse
from pathlib import Path
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

BASE_DIR = Path(__file__).resolve().parent


def build_bert4rec_overrides(raw_config: dict) -> dict:
    cd_cfg = raw_config.get("model_config", {})
    dataset_slug = slugify_dataset_name(raw_config["dataset"])
    ckpt_dir = BASE_DIR / "checkpoints" / "bert4rec" / dataset_slug

    return {
        # 复用你现有通用训练配置
        "data_path": raw_config["data_path"],
        "device": raw_config["device"],
        "seed": raw_config["seed"],
        "reproducibility": True,
        "epochs": raw_config["epochs"],
        "train_batch_size": raw_config["train_batch_size"],
        "eval_batch_size": raw_config["eval_batch_size"],
        "learning_rate": raw_config["learning_rate"],
        "weight_decay": raw_config["weight_decay"],
        "topk": raw_config["topk"],
        "metrics": ["Recall", "NDCG"],
        "valid_metric": "NDCG@10",
        "eval_step": 1,
        "stopping_step": 10,
        "train_neg_sample_args": None,   # BERT4Rec + CE 不需要负采样
        "checkpoint_dir": str(ckpt_dir),
        "show_progress": True,

        # 把你现有 CD_GRM 的部分超参映射到 BERT4Rec
        "hidden_size": cd_cfg.get("embed_dim", 128),
        "inner_size": cd_cfg.get("embed_dim", 128) * 4,
        "n_layers": cd_cfg.get("transformer_layers", 2),
        "n_heads": cd_cfg.get("nhead", 4),

        # BERT4Rec 自己的关键参数
        "mask_ratio": 0.2,
        "loss_type": "CE",
        "hidden_dropout_prob": 0.2,
        "attn_dropout_prob": 0.2,
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-12,
        "initializer_range": 0.02,
    }


def main() -> None:
    args = build_common_parser()

    config_path = resolve_config_path(args.config)
    raw_config = load_raw_config(config_path)

    data_file = resolve_data_path(raw_config)
    device = resolve_device(raw_config, args.device)
    set_global_seed(raw_config["seed"])

    bert4rec_overrides = build_bert4rec_overrides(raw_config)

    print("=" * 60)
    print(f"Training BERT4Rec on {raw_config['dataset']}")
    print(f"Config    : {config_path}")
    print(f"Data file : {data_file}")
    print(f"Device    : {device}")
    print(f"Save dir  : {bert4rec_overrides['checkpoint_dir']}")
    print("=" * 60)

    run_recbole(
        model="BERT4Rec",
        dataset=raw_config["dataset"],
        config_file_list=[str(config_path)],
        config_dict=bert4rec_overrides,
        saved=True,
    )


if __name__ == "__main__":
    main()
