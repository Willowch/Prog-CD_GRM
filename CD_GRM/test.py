from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import warnings

import optuna
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger

from CD_GRM.cd_grm.cd_grm_model import CD_GRM_Model
from CD_GRM.cd_grm.metrics.recbole_eval import evaluate_fullsort_recbole
from CD_GRM.utils import (
    load_raw_config,
    parse_test_args,
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


def build_runtime_meta(dataset_name: str) -> dict:
    dataset_slug = slugify_dataset_name(dataset_name)
    db_path = BASE_DIR / f"cd_grm_{dataset_slug}_tuning.db"
    ckpt_root = BASE_DIR / "checkpoints" / "optuna" / dataset_slug
    return {
        "dataset_slug": dataset_slug,
        "study_name": f"CD_GRM-{dataset_name}-Tuning",
        "storage_uri": f"sqlite:///{db_path.as_posix()}",
        "db_path": db_path,
        "ckpt_root": ckpt_root,
    }


def list_completed_trials(runtime_meta: dict) -> list[int]:
    if not runtime_meta["db_path"].exists():
        return []

    study = optuna.load_study(
        study_name=runtime_meta["study_name"],
        storage=runtime_meta["storage_uri"],
    )

    completed_trials = []
    for study_trial in study.trials:
        if study_trial.state != optuna.trial.TrialState.COMPLETE:
            continue

        checkpoint_path = runtime_meta["ckpt_root"] / f"trial_{study_trial.number}" / "best_model.pth"
        if checkpoint_path.exists():
            completed_trials.append(study_trial.number)
    return completed_trials


def locate_checkpoint(args: argparse.Namespace, runtime_meta: dict) -> tuple[Path, int | None]:
    if args.trial is not None:
        checkpoint_path = runtime_meta["ckpt_root"] / f"trial_{args.trial}" / "best_model.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path.resolve(), args.trial

    if not runtime_meta["db_path"].exists():
        raise FileNotFoundError(
            f"Optuna DB not found: {runtime_meta['db_path']} for best model.\n"
            f"Please pass --trial explicitly."
        )

    study = optuna.load_study(
        study_name=runtime_meta["study_name"],
        storage=runtime_meta["storage_uri"],
    )
    best_trial_num = study.best_trial.number
    checkpoint_path = runtime_meta["ckpt_root"] / f"trial_{best_trial_num}" / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Best trial is {best_trial_num}, but checkpoint is missing: {checkpoint_path}"
        )
    return checkpoint_path.resolve(), best_trial_num


def apply_trial_params(raw_config: dict, runtime_meta: dict, trial_num: int | None) -> dict:
    if trial_num is None or not runtime_meta["db_path"].exists():
        return {}

    study = optuna.load_study(
        study_name=runtime_meta["study_name"],
        storage=runtime_meta["storage_uri"],
    )
    trial = next((study_trial for study_trial in study.trials if study_trial.number == trial_num), None)
    if trial is None:
        raise ValueError(f"Trial {trial_num} not found in study {runtime_meta['study_name']}.")

    applied_params = {}
    model_config = raw_config.get("model_config", {})
    for key, value in trial.params.items():
        if key in model_config:
            model_config[key] = value
            applied_params[key] = value
        elif key in raw_config:
            raw_config[key] = value
            applied_params[key] = value
    return applied_params


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_item_loss_weight = 0.0

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        checkpoint_item_loss_weight = float(checkpoint.get("item_loss_weight", 0.0))
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(f"Unexpected checkpoint format: {type(checkpoint)}")

    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict, strict=True)

    if hasattr(model, "engine"):
        model.engine.item_loss_weight = checkpoint_item_loss_weight
        model.engine.use_item_ranker = checkpoint_item_loss_weight > 0


def build_model_for_trial(
    trial_num: int,
    raw_config: dict,
    runtime_meta: dict,
    train_data,
    device: torch.device,
):
    trial_config = deepcopy(raw_config)
    checkpoint_path = runtime_meta["ckpt_root"] / f"trial_{trial_num}" / "best_model.pth"
    applied_params = apply_trial_params(trial_config, runtime_meta, trial_num)
    trial_config["model_config"]["max_degree"] = min(
        trial_config["model_config"]["max_degree"],
        int(train_data.dataset.item_num) - 1,
    )

    model = CD_GRM_Model(trial_config, train_data.dataset).to(device)
    load_checkpoint(model, checkpoint_path, device)
    return model, checkpoint_path, applied_params, trial_config


def evaluate_trial_on_valid(
    trial_num: int,
    raw_config: dict,
    runtime_meta: dict,
    train_data,
    valid_data,
    rb_config,
    device: torch.device,
):
    model, checkpoint_path, applied_params, trial_config = build_model_for_trial(
        trial_num=trial_num,
        raw_config=raw_config,
        runtime_meta=runtime_meta,
        train_data=train_data,
        device=device,
    )
    valid_metrics = evaluate_fullsort_recbole(model, valid_data, rb_config, device)
    valid_score = float(valid_metrics["ndcg@10"])
    return checkpoint_path, applied_params, trial_config, valid_metrics, valid_score


def select_best_valid_trial(raw_config: dict, runtime_meta: dict, train_data, valid_data, rb_config, device):
    candidate_trials = list_completed_trials(runtime_meta)
    if not candidate_trials:
        return None

    print("=" * 60)
    print(f"Auto-selecting best valid trial from completed trials: {candidate_trials}")
    print("=" * 60)

    selected_trial_bundle = None
    best_valid_score = -float("inf")

    for candidate_trial in candidate_trials:
        checkpoint_path = runtime_meta["ckpt_root"] / f"trial_{candidate_trial}" / "best_model.pth"
        print(f"\n[Trial {candidate_trial}] Checkpoint : {checkpoint_path}")
        try:
            applied_params, trial_config, valid_metrics, valid_score = None, None, None, None
            checkpoint_path, applied_params, trial_config, valid_metrics, valid_score = evaluate_trial_on_valid(
                trial_num=candidate_trial,
                raw_config=raw_config,
                runtime_meta=runtime_meta,
                train_data=train_data,
                valid_data=valid_data,
                rb_config=rb_config,
                device=device,
            )
        except RuntimeError:
            print(f"[Trial {candidate_trial}] Skipped incompatible checkpoint: architecture mismatch.")
            continue

        if applied_params:
            print(f"[Trial {candidate_trial}] Params      : {applied_params}")
        print(f"[Trial {candidate_trial}] Valid ndcg@10: {valid_score:.4f}")

        if valid_score > best_valid_score:
            best_valid_score = valid_score
            selected_trial_bundle = {
                "trial_num": candidate_trial,
                "checkpoint_path": checkpoint_path,
                "applied_params": applied_params,
                "trial_config": trial_config,
                "valid_metrics": valid_metrics,
            }

    return selected_trial_bundle


def print_metric_block(title: str, metrics: dict) -> None:
    print("\n" + "=" * 48)
    print(title)
    print("=" * 48)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name:>16}: {metric_value:.4f}")


def main() -> None:
    args = parse_test_args()
    config_path = resolve_config_path(args.config)
    raw_config = load_raw_config(config_path)
    data_root = resolve_data_path(raw_config)

    set_global_seed(raw_config["seed"])
    device = resolve_device(raw_config, args.device)
    runtime_meta = build_runtime_meta(raw_config["dataset"])

    rb_config = Config(
        model="SASRec",
        dataset=raw_config["dataset"],
        config_dict=raw_config,
    )
    init_logger(rb_config)

    dataset = create_dataset(rb_config)
    train_data, valid_data, test_data = data_preparation(rb_config, dataset)

    raw_config["model_config"]["max_degree"] = min(
        raw_config["model_config"]["max_degree"],
        int(train_data.dataset.item_num) - 1,
    )

    selected_trial_bundle = None
    auto_select_best_trial = args.trial is None and raw_config.get("auto_select_best_trial", True)
    if auto_select_best_trial:
        selected_trial_bundle = select_best_valid_trial(
            raw_config=raw_config,
            runtime_meta=runtime_meta,
            train_data=train_data,
            valid_data=valid_data,
            rb_config=rb_config,
            device=device,
        )

    if selected_trial_bundle is None:
        checkpoint_path, trial_num = locate_checkpoint(args, runtime_meta)
        applied_params = apply_trial_params(raw_config, runtime_meta, trial_num)
        selected_trial_bundle = {
            "trial_num": trial_num,
            "checkpoint_path": checkpoint_path,
            "applied_params": applied_params,
            "trial_config": raw_config,
        }

    checkpoint_path = selected_trial_bundle["checkpoint_path"]
    trial_num = selected_trial_bundle["trial_num"]
    applied_params = selected_trial_bundle["applied_params"]
    eval_config = selected_trial_bundle["trial_config"]

    print("=" * 60)
    print(f"Evaluating CD_GRM on {eval_config['dataset']}")
    print(f"Config     : {config_path}")
    print(f"Data root  : {data_root}")
    print(f"Checkpoint : {checkpoint_path}")
    if trial_num is not None:
        print(f"Trial      : {trial_num}")
    if applied_params:
        print(f"Trial Params: {applied_params}")
    print(f"Device     : {device}")
    print("=" * 60)

    model = CD_GRM_Model(eval_config, train_data.dataset).to(device)
    load_checkpoint(model, checkpoint_path, device)
    valid_metrics = evaluate_fullsort_recbole(model, valid_data, rb_config, device)
    test_metrics = evaluate_fullsort_recbole(model, test_data, rb_config, device)

    print_metric_block("CD_GRM Valid Performance (RecBole Full-Sort)", valid_metrics)
    print_metric_block("CD_GRM Test Performance (RecBole Full-Sort)", test_metrics)
    print("=" * 48)
    print("Evaluation completed.")


if __name__ == "__main__":
    main()
