from __future__ import annotations

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
import warnings

warnings.filterwarnings(
    "ignore",
    message="A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.*",
    category=FutureWarning,
)

BASE_DIR = Path(__file__).resolve().parent


def resolve_bert4rec_config_path(config_arg: str) -> Path:
    candidate = Path(config_arg)
    if candidate.exists():
        return candidate.resolve()

    config_map = {
        "all_beauty": Path("CD_GRM/config/bert4rec_all_beauty.yaml"),
        "sports_and_outdoors": Path("CD_GRM/config/bert4rec_sports_and_outdoors.yaml"),
    }

    key = config_arg.strip().lower()
    config_path = config_map.get(key)
    if config_path is not None:
        if not config_path.exists():
            raise FileNotFoundError(f"BERT4Rec config file not found: {config_path}")
        return config_path.resolve()

    return resolve_config_path(config_arg)


def build_bert4rec_overrides(raw_config: dict) -> dict:
    cd_cfg = raw_config.get("model_config", {})
    dataset_slug = slugify_dataset_name(raw_config["dataset"])
    ckpt_dir = BASE_DIR / "checkpoints" / "bert4rec" / dataset_slug
    hidden_size = raw_config.get("hidden_size", cd_cfg.get("embed_dim", 128))
    inner_size = raw_config.get("inner_size", hidden_size * 4)

    return {
        "data_path": raw_config["data_path"],
        "device": raw_config["device"],
        "seed": raw_config["seed"],
        "reproducibility": raw_config.get("reproducibility", True),
        "epochs": raw_config.get("epochs", 50),
        "train_batch_size": raw_config.get("train_batch_size", 256),
        "eval_batch_size": raw_config.get("eval_batch_size", 64),
        "learning_rate": raw_config.get("learning_rate", 1e-3),
        "weight_decay": raw_config.get("weight_decay", 1e-4),
        "topk": raw_config.get("topk", [5, 10, 20]),
        "metrics": raw_config.get("metrics", ["Recall", "NDCG"]),
        "valid_metric": raw_config.get("valid_metric", "NDCG@10"),
        "eval_step": raw_config.get("eval_step", 1),
        "stopping_step": raw_config.get("stopping_step", 10),
        "train_neg_sample_args": raw_config.get("train_neg_sample_args"),
        "checkpoint_dir": str(ckpt_dir),
        "show_progress": raw_config.get("show_progress", True),
        "hidden_size": hidden_size,
        "inner_size": inner_size,
        "n_layers": raw_config.get("n_layers", cd_cfg.get("transformer_layers", 2)),
        "n_heads": raw_config.get("n_heads", cd_cfg.get("nhead", 4)),
        "mask_ratio": raw_config.get("mask_ratio", 0.2),
        "loss_type": raw_config.get("loss_type", "CE"),
        "hidden_dropout_prob": raw_config.get("hidden_dropout_prob", 0.2),
        "attn_dropout_prob": raw_config.get("attn_dropout_prob", 0.2),
        "hidden_act": raw_config.get("hidden_act", "gelu"),
        "layer_norm_eps": raw_config.get("layer_norm_eps", 1e-12),
        "initializer_range": raw_config.get("initializer_range", 0.02),
    }


def main() -> None:
    args = build_common_parser("Train BERT4Rec.").parse_args()

    config_path = resolve_bert4rec_config_path(args.config)
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
