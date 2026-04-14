from __future__ import annotations
import argparse
import sys
from pathlib import Path
import optuna
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent #当前脚本所在目录的绝对路径

from CD_GRM.utils import (
    load_raw_config,
    parse_test_args,
    resolve_config_path,
    resolve_data_path,
    resolve_device,
    set_global_seed,
    slugify_dataset_name,
)
from CD_GRM.cd_grm.metrics.evaluator import TopKEvaluator  #导入评估器
from CD_GRM.cd_grm.cd_grm_model import CD_GRM_Model  #导入CD_GRM


#元数据处理
def build_runtime_meta(dataset_name: str) -> dict:
    dataset_slug = slugify_dataset_name(dataset_name)  # 将数据集名称转成适合做文件名/目录名的 slug 形式
    db_path = BASE_DIR / f"cd_grm_{dataset_slug}_tuning.db"  # 该数据集对应的 Optuna SQLite 数据库路径
    ckpt_root = BASE_DIR / "checkpoints" / "optuna" / dataset_slug  # 该数据集对应的 Optuna checkpoint 根目录

    return {  # 返回一个运行时元信息字典
        "dataset_slug": dataset_slug,  # 数据集 slug
        "study_name": f"CD_GRM-{dataset_name}-Tuning",  # Optuna study 名称
        "storage_uri": f"sqlite:///{db_path.as_posix()}",  # Optuna 使用的 sqlite URI
        "db_path": db_path,  # 本地 sqlite 文件路径
        "ckpt_root": ckpt_root,  # checkpoint 根目录
    }
#找到最优模型路径
def locate_checkpoint(args: argparse.Namespace, runtime_meta: dict) -> tuple[Path, int | None]:
    if args.trial is not None:  # 如果没有指定 checkpoint，但指定了 trial 编号
        checkpoint_path = runtime_meta["ckpt_root"] / f"trial_{args.trial}" / "best_model.pth"  # 按约定拼出该 trial 的最佳模型路径
        if not checkpoint_path.exists():  # 如果该 trial 的模型文件不存在
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")  # 抛出异常
        return checkpoint_path.resolve(), args.trial  # 返回 checkpoint 路径和 trial 编号

    if not runtime_meta["db_path"].exists():  # 如果既没有指定 trial，并且 Optuna DB 文件也不存在
        raise FileNotFoundError(
            f"Optuna DB not found: {runtime_meta['db_path']}for best model\n"  # 提示找不到 Optuna 数据库
            f"Please pass --checkpoint or --trial explicitly."  # 提示用户显式传入 checkpoint 或 trial
        )

    study = optuna.load_study(  # 加载已有的 Optuna study
        study_name=runtime_meta["study_name"],  # 指定 study 名称
        storage=runtime_meta["storage_uri"],  # 指定存储位置
    )

    best_trial_num = study.best_trial.number  # 获取最优 trial 的编号
    checkpoint_path = runtime_meta["ckpt_root"] / f"trial_{best_trial_num}" / "best_model.pth"  # 构造最优 trial 的模型文件路径

    if not checkpoint_path.exists():  # 如果理论上的“最优 trial 模型文件”并不存在
        raise FileNotFoundError(
            f"Best trial is {best_trial_num}, but checkpoint is missing: {checkpoint_path}"  # 报错说明数据库里有最优 trial，但模型文件没了
        )

    return checkpoint_path.resolve(), best_trial_num  # 返回最终 checkpoint 路径和最优 trial 编号
#模型加载最优权重 清空推理树
def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)  # 从磁盘加载 checkpoint，并映射到指定设备

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:  # 如果 checkpoint 是字典，且包含 "state_dict" 键
        state_dict = checkpoint["state_dict"]  # 取出真正的参数字典
    else:
        state_dict = checkpoint  # 否则认为 checkpoint 本身就是 state_dict

    if not isinstance(state_dict, dict):  # 如果得到的结果不是参数字典
        raise TypeError(f"Unexpected checkpoint format: {type(checkpoint)}")  # 抛出类型异常

    if any(key.startswith("module.") for key in state_dict):  # 如果参数名中带有 "module." 前缀，通常说明来自 DataParallel / DDP
        state_dict = {
            key.removeprefix("module."): value  # 去掉前缀 "module."
            for key, value in state_dict.items()  # 遍历所有参数
        }

    model.load_state_dict(state_dict, strict=True)  # 严格加载模型参数，要求参数名与形状完全匹配

    if hasattr(model, "infer_engine") and hasattr(model.infer_engine, "invalidate_trie_cache"):  # 如果模型的推理引擎支持失效 Trie 缓存
        model.infer_engine.invalidate_trie_cache()  # 加载新参数后，主动清空旧 Trie，防止使用过期映射
#更新用户历史
def _update_user_history(
    user_history_dict: dict[int, set[int]],  # 用户历史字典：user_id -> 该用户交互过的 item 集合
    dataloader,  # 某个数据加载器（train / valid / test）
    include_target: bool,  # 是否把当前样本中的 target item 也纳入历史
) -> None:
    if dataloader is None:  # 如果 dataloader 为空
        return  # 直接返回，不做任何处理

    dataset = dataloader.dataset  # 取出 dataloader 对应的数据集对象
    users = dataset.inter_feat["user_id"].cpu().numpy()  # 取出所有 user_id，并转到 CPU 再转成 numpy
    item_seqs = dataset.inter_feat["item_id_list"].cpu().numpy()  # 取出所有历史序列 item_id_list
    targets = dataset.inter_feat["item_id"].cpu().numpy()  # 取出每条样本对应的 target item

    for user_id, item_seq, target_item in zip(users, item_seqs, targets):  # 逐条样本遍历用户、历史序列和目标 item
        user_id = int(user_id)  # 把用户 id 转成 Python int
        history = user_history_dict.setdefault(user_id, set())  # 如果该用户还没有历史集合，则初始化为空 set，并返回该 set

        history.update(int(item_id) for item_id in item_seq.tolist() if int(item_id) != 0)  # 把历史序列中非 PAD(0) 的 item 全部加入该用户历史集合

        if include_target:  # 如果当前阶段需要把 target 也算入历史
            target_item = int(target_item)  # 把目标 item 转成 Python int
            if target_item != 0:  # 如果目标 item 不是 PAD/空值
                history.add(target_item)  # 把目标 item 也加入该用户历史集合
#根据训练，验证，测试集构建 用户历史字典
def build_user_history_dict(train_data, valid_data, test_data) -> dict[int, set[int]]:
    user_history_dict: dict[int, set[int]] = {}  # 初始化“用户 -> 历史 item 集合”的字典

    # 训练集和验证集中的 target，在最终测试目标发生前，都已经属于用户历史的一部分
    _update_user_history(user_history_dict, train_data, include_target=True)  # 将训练集历史和 target 纳入用户历史
    _update_user_history(user_history_dict, valid_data, include_target=True)  # 将验证集历史和 target 纳入用户历史

    # 测试集自己的目标 item 是当前要预测的对象，不能提前过滤掉
    _update_user_history(user_history_dict, test_data, include_target=False)  # 仅加入测试集中的历史序列，不加入测试 target

    return user_history_dict  # 返回构建完成的用户历史字典
#构建所有物品的sid
def build_current_valid_sid_pool(model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    model.eval()  # 切换模型到评估模式，关闭 dropout / batchnorm 的训练行为

    with torch.no_grad():  # 关闭梯度计算，节省显存并提升推理速度
        item_vocab_size = model.item_vocab_size  # 获取 item 词表大小
        all_item_ids = torch.arange(1, item_vocab_size, device=device)  # 构造所有真实 item 的 ID，跳过 0（通常 0 是 PAD）
        all_item_embs = model.engine.item_embedding(all_item_ids)  # 取出所有 item 的 embedding
        _, current_sids, _ = model.engine.quantizer(all_item_embs)  # 用量化器把 item embedding 映射成当前的 semantic IDs
        valid_sid_pool = current_sids + 1  # 所有 SID 加 1，通常是为了避开 0 这个 PAD 保留值

    return valid_sid_pool  # 返回当前所有合法 item 对应的 SID 池


def main() -> None:
    args = parse_test_args()
    config_path = resolve_config_path(args.config)  # 解析配置文件路径;只支持小写原名
    raw_config = load_raw_config(config_path)  # 读取 YAML 原始配置内容
    data_root = resolve_data_path(raw_config)  # 规范化并确认数据集根目录，同时写回 raw_config["data_path"]

    set_global_seed(raw_config["seed"])  # 设置随机种子，保证实验可复现
    device = resolve_device(raw_config, args.device)  # 根据命令行和配置确定最终运行设备
    runtime_meta = build_runtime_meta(raw_config["dataset"])  # 根据数据集名称构造运行时元信息
    checkpoint_path, trial_num = locate_checkpoint(args, runtime_meta)  # 确定最终用于评估的 checkpoint 路径，以及可选的 trial 编号

    rb_config = Config(  # 创建 RecBole 配置对象
        model="SASRec",  # 这里借用 SASRec 的配置范式来驱动 RecBole 数据流程
        dataset=raw_config["dataset"],  # 数据集名称
        config_dict=raw_config,  # 传入原始配置字典
    )
    init_logger(rb_config)  # 初始化日志系统

    print("=" * 60)
    print(f"Evaluating CD_GRM on {raw_config['dataset']}")  # 打印当前评估的数据集
    print(f"Config     : {config_path}")  # 打印配置文件路径
    print(f"Data root  : {data_root}")  # 打印数据根目录
    print(f"Checkpoint : {checkpoint_path}")  # 打印 checkpoint 路径
    if trial_num is not None:  # 如果当前 checkpoint 对应某个具体 trial
        print(f"Trial      : {trial_num}")  # 打印 trial 编号
    print(f"Device     : {device}")  # 打印运行设备
    print("=" * 60)  # 打印分隔线

    dataset = create_dataset(rb_config)  # 使用 RecBole 创建数据集对象
    train_data, valid_data, test_data = data_preparation(rb_config, dataset)  # 按 RecBole 规则准备 train / valid / test dataloader
    user_history_dict = build_user_history_dict(train_data, valid_data, test_data)  # 构造每个用户的完整历史交互集合

    model = CD_GRM_Model(raw_config, train_data.dataset).to(device)  # 创建 CD_GRM 模型，并移动到指定设备
    load_checkpoint(model, checkpoint_path, device)  # 加载训练好的模型参数

    valid_sid_pool = build_current_valid_sid_pool(model, device)  # 基于当前模型参数构建合法 SID 池

    evaluator = TopKEvaluator(  # 创建 Top-K 排序评估器
        k_list=raw_config["topk"],  # 需要评估的 K 值列表，例如 [5, 10, 20]
        codebook_size=raw_config["model_config"]["codebook_size"],  # 模型码本大小
        m_layers=raw_config["model_config"]["m_layers"],  # Semantic ID 的层数
    ).cpu()  # 将评估器放到 CPU 上，避免不必要的 GPU 占用

    all_preds = []  # 用于收集所有 batch 的 Top-K 预测结果
    all_targets = []  # 用于收集所有 batch 的真实 target item
    all_gen_sids = []  # 用于收集所有 batch 的生成 SID
    max_k = max(raw_config["topk"])  # 获取评估中最大的 K，用于一次性生成足够多的候选

    model.eval()  # 切换模型到评估模式
    with torch.no_grad():  # 关闭梯度计算
        for batch_tuple in tqdm(test_data, desc="Inference"):  # 遍历测试集，并显示“Inference”进度条
            batch_data = batch_tuple[0] if isinstance(batch_tuple, (tuple, list)) else batch_tuple  # 兼容 dataloader 返回 tuple/list 或直接返回 batch 的两种情况

            history_seq = batch_data["item_id_list"].to(device)  # 取出用户历史序列，并移动到设备上
            target_item = batch_data["item_id"].to(device)  # 取出真实目标 item，并移动到设备上
            user_ids = batch_data["user_id"].cpu().numpy()  # 取出用户 ID，转到 CPU 并转为 numpy

            batch_full_history = [  # 为当前 batch 中每个用户构造完整历史列表
                sorted(user_history_dict.get(int(user_id), set()))  # 从历史字典中取出该用户已交互过的 item 集合，并排序成列表
                for user_id in user_ids  # 遍历当前 batch 的所有用户
            ]

            topk_preds, gen_sids = model.predict_topk(  # 调用模型的 Top-K 预测接口
                history_seq,  # 输入用户历史序列
                top_k=max_k,  # 生成最大的 K 个候选
                full_history_list=batch_full_history,  # 传入完整历史，用于过滤已见 item
            )

            all_preds.append(topk_preds.cpu())  # 当前 batch 的预测结果转到 CPU 后保存
            all_targets.append(target_item.cpu())  # 当前 batch 的真实标签转到 CPU 后保存
            all_gen_sids.append(gen_sids.cpu())  # 当前 batch 的生成 SID 转到 CPU 后保存

    all_preds_tensor = torch.cat(all_preds, dim=0)  # 将所有 batch 的预测结果沿 batch 维拼接
    all_targets_tensor = torch.cat(all_targets, dim=0)  # 将所有 batch 的真实标签拼接
    all_gen_sids_tensor = torch.cat(all_gen_sids, dim=0)  # 将所有 batch 的生成 SID 拼接

    metrics = evaluator.evaluate_ranking(all_preds_tensor, all_targets_tensor)  # 计算 HR/NDCG 等排序指标
    metrics["IGR (Invalid Gen Rate)"] =evaluator.evaluate_igr(all_gen_sids_tensor,valid_sid_pool.cpu())

    print("\n" + "=" * 40)  # 打印结果区域分隔线
    print("CD_GRM Final Test Performance")  # 打印结果标题
    print("=" * 40)  # 打印分隔线
    for metric_name, metric_value in metrics.items():  # 遍历所有评估指标
        if "IGR" in metric_name:  # 如果是 IGR 指标
            print(f"{metric_name:>22}: {metric_value * 100:.2f}% (Lower is better)")  # 按百分比格式打印，且说明越低越好
        else:  # 如果是普通排序指标
            print(f"{metric_name:>22}: {metric_value:.4f}")  # 按四位小数打印
    print("=" * 40)  # 打印分隔线
    print("Evaluation completed.")  # 打印评估完成提示

if __name__ == "__main__":  # 如果当前文件是作为主程序直接运行
    main()  # 执行主函数