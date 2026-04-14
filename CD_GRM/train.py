from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import optuna
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation

from CD_GRM.utils import (
    load_raw_config,
    parse_train_args,
    resolve_config_path,
    set_global_seed,
    slugify_dataset_name,
    resolve_device
)

from CD_GRM.cd_grm.trainer import AMPTrainer
from CD_GRM.cd_grm.cd_grm_model import CD_GRM_Model


BASE_DIR = Path(__file__).resolve().parent

def build_runtime_meta(dataset_name: str) -> dict:
    dataset_slug = slugify_dataset_name(dataset_name) #数据集名称转小写
    run_root = BASE_DIR / "runs" / "optuna" / dataset_slug #构造该数据集对应的Optuna运行日志目录
    ckpt_root = BASE_DIR / "checkpoints" / "optuna" / dataset_slug #构造该数据集对应的 checkpoint保存目录
    db_path = BASE_DIR / f"cd_grm_{dataset_slug}_tuning.db" #构造该数据集对应的SQLite数据库文件路径

    return {
        "dataset_slug": dataset_slug,# 数据集的slug名
        "study_name": f"CD_GRM-{dataset_name}-Tuning", #Optuna study名称例如 CD_GRM-All_Beauty-Tuning
        "storage_uri": f"sqlite:///{db_path.as_posix()}",#Optuna 使用的数据库连接 URI as_posix() 把路径转成 / 风格，便于 URI 使用
        "run_root": run_root,#日志目录根路径
        "ckpt_root": ckpt_root,# checkpoint根路径
        "db_path": db_path,#数据库文件路径
    }

def make_objective(base_raw_config: dict, runtime_meta: dict,args: argparse.Namespace):

    def objective(trial: optuna.trial.Trial) -> float:

        raw_config = deepcopy(base_raw_config)# 对基础配置做深拷贝 这样每个 trial 的参数修改都不会污染原始配置
        raw_config["learning_rate"] = trial.suggest_float(
            "learning_rate",
            1e-4,#搜索下界
            5e-3,#搜索上界
            log=True,#采用对数尺度采样，适合学习率这种跨数量级参数
        )

        raw_config["model_config"]["tau"] = trial.suggest_float(
            "tau",#超参数名tau
            0.05,#搜索下界
            0.2,#搜索上界
        )

        raw_config["model_config"]["diffusion_steps"] = trial.suggest_categorical(
            "diffusion_steps",#超参数名diffusion_steps
            [10, 20, 50],#从这几个离散候选值中选一个
        )

        raw_config["train_batch_size"] = trial.suggest_categorical(
            "train_batch_size",#超参数名：train_batch_size
            [256, 512],
        )

        set_global_seed(raw_config["seed"])

        #dataloader准备
        rb_config = Config(
            model="SASRec",#构造RecBole配置时指定 model="SASRec"这里通常是借用 RecBole 的序列数据处理管线，不一定真的训练 SASRec
            dataset=raw_config["dataset"],#数据集名称，从配置中取出
            config_dict=raw_config,#把修改后的原始配置字典整体传给 RecBole
        )
        print(f"\n[Trial {trial.number}] Building dataset: {raw_config['dataset']}")#打印当前 trial 编号和正在构建的数据集名称
        dataset = create_dataset(rb_config)
        train_data, valid_data, _ = data_preparation(rb_config, dataset)
        device = resolve_device(raw_config, args.device)
        #模型准备
        model = CD_GRM_Model(
            raw_config,# 把当前 trial 使用的配置传给模型
            train_data.dataset,# 把训练 dataloader 内部的数据集对象传给模型
        ).to(device)

        trial_log_dir = runtime_meta["run_root"] / f"trial_{trial.number}"# 当前 trial 的日志目录
        trial_ckpt_dir = runtime_meta["ckpt_root"] / f"trial_{trial.number}"# 当前 trial 的 checkpoint 保存目录

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
        best_score = trainer.train() #验证集上的最佳分数
        return best_score #把当前 trial 的最佳分数返回给 Optuna

    return objective
    #返回内部定义好的 objective 函数
    #这样 main 中就能把它交给 study.optimize 使用


def main() -> None:

    args = parse_train_args()

    config_path = resolve_config_path(args.config)# 根据命令行给的 config 参数解析出真实存在的配置文件路径
    base_raw_config = load_raw_config(config_path)# 读取 YAML 配置文件，得到基础配置字典
    runtime_meta = build_runtime_meta(base_raw_config["dataset"])# 根据数据集名称构建运行时元信息

    print("=" * 60)
    print(f"Starting Optuna Hyperparameter Optimization on {base_raw_config['dataset']}...")
    print(f"Config : {config_path}")# 打印配置文件路径
    print(f"Study  : {runtime_meta['study_name']}")# 打印当前 study 名称
    print(f"DB     : {runtime_meta['db_path']}")# 打印当前 Optuna 使用的 SQLite 数据库文件路径
    print("=" * 60)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,#前5个trial不剪枝，先积累对比基线
        n_warmup_steps=10,# 每个 trial 前 10 个汇报 step 不剪枝
        interval_steps=1,# 每隔 1 个 step 检查一次是否需要剪枝
    )

    study = optuna.create_study(
        direction="maximize",# 优化目标是“越大越好”，例如最大化 NDCG/HR 等验证指标
        study_name=runtime_meta["study_name"],# 当前 study 名称
        storage=runtime_meta["storage_uri"],# 使用 SQLite 数据库存储调参过程和结果
        load_if_exists=True,# 如果这个 study 已存在，则继续使用，不重新创建
        pruner=pruner,# 指定当前 study 使用的剪枝器
    )

    objective = make_objective(base_raw_config, runtime_meta,args)# 用基础配置和运行时元信息构造 objective 函数

    study.optimize(
        objective,# 指定每个 trial 要执行的目标函数
        n_trials=args.n_trials,# 指定总 trial 数，由命令行参数控制
    )
    print("\n" + "=" * 60)
    print("Optimization Finished!")
    print("=" * 60)

    pruned_trials = study.get_trials(
        deepcopy=False,# 不做深拷贝，减少内存开销
        states=[optuna.trial.TrialState.PRUNED],# 只取被剪枝停止的 trial
    )

    complete_trials = study.get_trials(
        deepcopy=False,
        states=[optuna.trial.TrialState.COMPLETE],# 只取正常完成的 trial
    )

    print(f"Total trials   : {len(study.trials)}")#打印总共执行过的 trial 数量
    print(f"Pruned trials  : {len(pruned_trials)}")#打印被剪枝的 trial 数量
    print(f"Complete trials: {len(complete_trials)}")#打印完整跑完的 trial 数量

    print("\nBest trial:") # 打印提示：下面输出最佳 trial 的信息
    trial = study.best_trial# 获取当前 study 中表现最好的那个 trial
    print(f"  Best validation score: {trial.value:.4f}")# 打印最佳 trial 的目标值，一般就是最佳验证分数

    print("  Best hyperparameters:")# 打印提示：下面输出最佳超参数组合
    for key, value in trial.params.items():
        print(f"    {key}: {value}")# 打印每个超参数名和对应取值

if __name__ == "__main__":
    # 这是 Python 脚本的标准入口判断：
    # 只有当前文件被直接运行时，下面的 main() 才会执行
    # 如果该文件被其他模块 import，则不会自动执行

    main()
    # 调用主函数，正式启动整个调参流程