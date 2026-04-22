from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
import warnings

import optuna
from recbole.config import Config
from recbole.data import create_dataset, data_preparation

from CD_GRM.cd_grm.cd_grm_model import CD_GRM_Model
from CD_GRM.cd_grm.trainer import AMPTrainer
from CD_GRM.utils import (
    load_raw_config,
    parse_train_args,
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
    run_root = BASE_DIR / "runs" / "optuna" / dataset_slug
    ckpt_root = BASE_DIR / "checkpoints" / "optuna" / dataset_slug
    db_path = BASE_DIR / f"cd_grm_{dataset_slug}_tuning.db"

    return {
        "dataset_slug": dataset_slug,
        "study_name": f"CD_GRM-{dataset_name}-Tuning",
        "storage_uri": f"sqlite:///{db_path.as_posix()}",
        "run_root": run_root,
        "ckpt_root": ckpt_root,
        "db_path": db_path,
    }


def make_objective(base_raw_config: dict, runtime_meta: dict, args):

    def objective(trial: optuna.trial.Trial) -> float:
        raw_config = deepcopy(base_raw_config)
        mcfg = raw_config["model_config"]

        raw_config["train_batch_size"] = trial.suggest_categorical(
            "train_batch_size", [128]
        )

        raw_config["learning_rate"] = trial.suggest_float(
            "learning_rate",0.0006150745904964907, 0.0006150745904964907
        )
        raw_config["weight_decay"] = trial.suggest_float(
            "weight_decay",0.0009340308327072935, 0.0009340308327072935
        )

        mcfg["tau"] = trial.suggest_float(
            "tau", 0.057317870293182936, 0.057317870293182936
        )
        mcfg["diffusion_steps"] = trial.suggest_categorical(
            "diffusion_steps", [20]
        )
        mcfg["max_degree"] = trial.suggest_categorical(
            "max_degree", [7]
        )

        raw_config["stage1_epochs"] = trial.suggest_categorical(
            "stage1_epochs", [10]
        )
        raw_config["freeze_rqvae_stage2"] = trial.suggest_categorical(
            "freeze_rqvae_stage2", [True]
        )

        set_global_seed(raw_config["seed"])

        # Reuse RecBole's sequential data pipeline for dataset construction.
        rb_config = Config(
            model="SASRec",
            dataset=raw_config["dataset"],
            config_dict=raw_config,
        )
        print(f"\n[Trial {trial.number}] Building dataset: {raw_config['dataset']}")
        dataset = create_dataset(rb_config)
        train_data, valid_data, _ = data_preparation(rb_config, dataset)

        steps_per_epoch = len(train_data)
        warmup_steps = raw_config["model_config"].get("dead_code_warmup_steps", 100)
        min_stage1_epochs = math.ceil(warmup_steps / max(1, steps_per_epoch))
        raw_config["stage1_epochs"] = max(raw_config["stage1_epochs"], min_stage1_epochs)

        device = resolve_device(raw_config, args.device)
        raw_config["model_config"]["max_degree"] = min(
            raw_config["model_config"]["max_degree"],
            int(train_data.dataset.item_num) - 1,
        )

        model = CD_GRM_Model(
            raw_config,
            train_data.dataset,
        ).to(device)

        trial_log_dir = runtime_meta["run_root"] / f"trial_{trial.number}"
        trial_ckpt_dir = runtime_meta["ckpt_root"] / f"trial_{trial.number}"

        trainer = AMPTrainer(
            model=model,
            train_dataloader=train_data,
            valid_dataloader=valid_data,
            learning_rate=raw_config["learning_rate"],
            weight_decay=raw_config["weight_decay"],
            epochs=raw_config["epochs"],
            patience=5,
            device=device,
            log_dir=str(trial_log_dir),
            checkpoint_dir=str(trial_ckpt_dir),
            trial=trial,
            stage1_epochs=raw_config.get("stage1_epochs", 15),
            freeze_rqvae_stage2=raw_config.get("freeze_rqvae_stage2", True),
        )
        best_score = trainer.train()
        return best_score

    return objective


def main() -> None:
    args = parse_train_args()

    config_path = resolve_config_path(args.config)
    base_raw_config = load_raw_config(config_path)
    runtime_meta = build_runtime_meta(base_raw_config["dataset"])

    print("=" * 60)
    print(f"Starting Optuna Hyperparameter Optimization on {base_raw_config['dataset']}...")
    print(f"Config : {config_path}")
    print(f"Study  : {runtime_meta['study_name']}")
    print(f"DB     : {runtime_meta['db_path']}")
    print("=" * 60)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10,
        interval_steps=1,
    )

    sampler = optuna.samplers.TPESampler(
        seed=base_raw_config["seed"],
        multivariate=True,
        group=True,
        n_startup_trials=10,
    )

    study = optuna.create_study(
        direction="maximize",
        study_name=runtime_meta["study_name"],
        storage=runtime_meta["storage_uri"],
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    objective = make_objective(base_raw_config, runtime_meta, args)
    study.optimize(objective, n_trials=args.n_trials)

    print("\n" + "=" * 60)
    print("Optimization Finished!")
    print("=" * 60)

    pruned_trials = study.get_trials(
        deepcopy=False,
        states=[optuna.trial.TrialState.PRUNED],
    )
    complete_trials = study.get_trials(
        deepcopy=False,
        states=[optuna.trial.TrialState.COMPLETE],
    )

    print(f"Total trials   : {len(study.trials)}")
    print(f"Pruned trials  : {len(pruned_trials)}")
    print(f"Complete trials: {len(complete_trials)}")

    print("\nBest trial:")
    print(f"  Best validation score: {study.best_trial.value:.4f}")
    print("  Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
