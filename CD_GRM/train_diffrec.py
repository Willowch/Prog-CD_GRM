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


def resolve_diffrec_config_path(config_arg: str) -> Path:
    candidate = Path(config_arg)
    if candidate.exists():
        return candidate.resolve()

    config_map = {
        "all_beauty": Path("CD_GRM/config/diffrec_all_beauty.yaml"),
    }

    key = config_arg.strip().lower()
    config_path = config_map.get(key)
    if config_path is not None:
        if not config_path.exists():
            raise FileNotFoundError(f"DiffRec config file not found: {config_path}")
        return config_path.resolve()

    # Fallback: allow reusing the project's generic dataset config path resolver.
    return resolve_config_path(config_arg)


def build_diffrec_overrides(raw_config: dict) -> dict:
    dataset_slug = slugify_dataset_name(raw_config["dataset"])
    ckpt_dir = BASE_DIR / "checkpoints" / "diffrec" / dataset_slug

    return {
        # -------------------------
        # Runtime
        # -------------------------
        "data_path": raw_config["data_path"],
        "device": raw_config["device"],
        "seed": raw_config["seed"],
        "reproducibility": raw_config.get("reproducibility", True),
        "show_progress": raw_config.get("show_progress", True),
        "checkpoint_dir": str(ckpt_dir),

        # -------------------------
        # Training
        # Keep the official DiffRec optimizer style and a paper-like early-stop rhythm:
        # evaluate every 5 epochs, stop after 4 stale evaluations (= 20 epochs patience).
        # -------------------------
        "epochs": raw_config.get("epochs", 1000),
        "eval_step": raw_config.get("eval_step", 5),
        "stopping_step": raw_config.get("stopping_step", 4),
        "train_batch_size": raw_config.get("train_batch_size", 400),
        "eval_batch_size": raw_config.get("eval_batch_size", 400),
        "learner": raw_config.get("learner", "adamw"),
        "learning_rate": raw_config.get("learning_rate", 5e-5),
        "weight_decay": raw_config.get("weight_decay", 0.0),
        "train_neg_sample_args": None,

        # -------------------------
        # Evaluation
        # Keep the project's unified metric protocol.
        # -------------------------
        "topk": raw_config.get("topk", [5, 10, 20]),
        "metrics": raw_config.get("metrics", ["Recall", "NDCG"]),
        "valid_metric": raw_config.get("valid_metric", "NDCG@10"),

        # -------------------------
        # DiffRec parameters
        # Official amazon-book_clean strong setting, adapted only on dims_dnn.
        # -------------------------
        "noise_schedule": raw_config.get("noise_schedule", "linear-var"),
        "noise_scale": raw_config.get("noise_scale", 1e-4),
        "noise_min": raw_config.get("noise_min", 5e-4),
        "noise_max": raw_config.get("noise_max", 5e-3),
        "sampling_noise": raw_config.get("sampling_noise", False),
        "sampling_steps": raw_config.get("sampling_steps", 0),
        "reweight": raw_config.get("reweight", True),
        "mean_type": raw_config.get("mean_type", "x0"),
        "steps": raw_config.get("steps", 5),
        "history_num_per_term": raw_config.get("history_num_per_term", 10),
        "beta_fixed": raw_config.get("beta_fixed", True),

        "dims_dnn": raw_config.get("dims_dnn", [200]),
        "embedding_size": raw_config.get("embedding_size", 10),
        "mlp_act_func": raw_config.get("mlp_act_func", "tanh"),
        "norm": raw_config.get("norm", False),

        # Keep pure DiffRec by default. Set to true only if you want T-DiffRec.
        "time-aware": raw_config.get("time-aware", False),
        "w_max": raw_config.get("w_max", 1.0),
        "w_min": raw_config.get("w_min", 0.1),
    }


def main() -> None:
    args = build_common_parser("Train DiffRec.").parse_args()

    config_path = resolve_diffrec_config_path(args.config)
    raw_config = load_raw_config(config_path)

    data_file = resolve_data_path(raw_config)
    device = resolve_device(raw_config, args.device)
    set_global_seed(raw_config["seed"])

    diffrec_overrides = build_diffrec_overrides(raw_config)

    print("=" * 60)
    print(f"Training DiffRec on {raw_config['dataset']}")
    print(f"Config    : {config_path}")
    print(f"Data file : {data_file}")
    print(f"Device    : {device}")
    print(f"Save dir  : {diffrec_overrides['checkpoint_dir']}")
    print("=" * 60)

    run_recbole(
        model="DiffRec",
        dataset=raw_config["dataset"],
        config_file_list=[str(config_path)],
        config_dict=diffrec_overrides,
        saved=True,
    )


if __name__ == "__main__":
    main()
