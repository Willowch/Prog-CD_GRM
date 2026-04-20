import os
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import optuna
from CD_GRM.cd_grm.metrics.evaluator import TopKEvaluator
from CD_GRM.cd_grm.metrics.recbole_eval import evaluate_fullsort_recbole


class AMPTrainer:

    def __init__(self, model: nn.Module, train_dataloader, valid_dataloader=None,
                 learning_rate: float = 1e-3, weight_decay: float = 1e-4,
                 epochs: int = 50, patience: int = 5, device: torch.device = None,
                 log_dir: str = "CD_GRM/runs/cd_grm_experiment",
                 checkpoint_dir: str = "CD_GRM/checkpoints",
                 trial: optuna.trial.Trial = None,
                 stage1_epochs: int = 15,
                 freeze_rqvae_stage2: bool = True,
                 recbole_config=None):

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.stage1_epochs = stage1_epochs
        self.epochs = epochs
        self.patience = patience# 早停容忍
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay# 权重衰减系数
        self.freeze_rqvae_stage2 = freeze_rqvae_stage2#Stage2是否冻结 quantizer / RQ-VAE
        self.trial = trial# Optuna 的 trial 对象；若不是调参模式则为 None
        self.is_cuda = (self.device.type == "cuda")
        self.writer = SummaryWriter(log_dir=log_dir)# 创建 TensorBoard 日志记录器
        self.checkpoint_dir = checkpoint_dir# 保存 checkpoint 的目录路径

        os.makedirs(self.checkpoint_dir, exist_ok=True)#checkpoint目录不存在，就创建它；如果已存在则不报错
        self.best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")

        if self.valid_dataloader is not None:
            self.val_k = 10 #验证top-k固定为10
            self.evaluator = TopKEvaluator(
                k_list=[self.val_k],
                codebook_size=self.model.engine.quantizer.codebook_size,
                m_layers=self.model.engine.quantizer.num_layers,
            )

    def _configure_stage(self, stage: int):
        engine = self.model.engine

        # 先全部关掉，避免遗漏
        for _, param in engine.named_parameters():
            param.requires_grad = False

        if stage == 1:
            # Stage1: 只训练 item embedding；quantizer 通过 EMA buffer 更新
            for name, param in engine.named_parameters():
                if name.startswith("item_embedding."):
                    param.requires_grad = True

            engine.quantizer.freeze_codebook = False

        else:
            # Stage2: 训练 transformer / seq-ranker + log_vars
            for name, param in engine.named_parameters():
                if name.startswith("transformer."):
                    param.requires_grad = True
                elif name.startswith("seq_ranker."):
                    param.requires_grad = True
                elif name.startswith("item_embedding.") and getattr(engine, "finetune_item_embedding_stage2", False):
                    param.requires_grad = True
                elif name == "log_vars":
                    param.requires_grad = True

            if not self.freeze_rqvae_stage2:
                raise ValueError("Fixed-target stage2 requires freeze_rqvae_stage2=True.")

            engine.quantizer.freeze_codebook = True

    def _split_decay_params(self, named_params):
        decay_params = []
        no_decay_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue

            is_no_decay = (
                    name.endswith(".bias")
                    or "norm" in name.lower()
                    or name == "log_vars"
            )

            if is_no_decay:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return decay_params, no_decay_params

    def _build_stage_param_groups(self, stage: int):
        engine = self.model.engine
        named_params = list(engine.named_parameters())

        item_named = [
            (n, p) for n, p in named_params
            if n.startswith("item_embedding.") and p.requires_grad
        ]
        transformer_named = [
            (n, p) for n, p in named_params
            if n.startswith("transformer.") and p.requires_grad
        ]
        seq_ranker_named = [
            (n, p) for n, p in named_params
            if n.startswith("seq_ranker.") and p.requires_grad
        ]
        logvar_named = [
            (n, p) for n, p in named_params
            if n == "log_vars" and p.requires_grad
        ]

        groups = []

        def add_group(named_list, lr, weight_decay):
            if not named_list:
                return
            decay_params, no_decay_params = self._split_decay_params(named_list)
            if decay_params:
                groups.append({
                    "params": decay_params,
                    "lr": lr,
                    "weight_decay": weight_decay,
                })
            if no_decay_params:
                groups.append({
                    "params": no_decay_params,
                    "lr": lr,
                    "weight_decay": 0.0,
                })

        if stage == 1:
            add_group(item_named, lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            item_lr_scale = getattr(engine, "item_embedding_stage2_lr_scale", 0.5)
            add_group(item_named, lr=self.learning_rate * item_lr_scale, weight_decay=self.weight_decay)
            add_group(transformer_named, lr=self.learning_rate, weight_decay=self.weight_decay)
            add_group(seq_ranker_named, lr=self.learning_rate, weight_decay=self.weight_decay)

            if logvar_named:
                groups.append({
                    "params": [p for _, p in logvar_named],
                    "lr": self.learning_rate * 0.05,
                    "weight_decay": 0.0,
                })

        return groups

    def _setup_optimizer_and_scheduler(self, stage: int):
        self._configure_stage(stage)

        param_groups = self._build_stage_param_groups(stage)
        if not param_groups:
            raise RuntimeError(f"No trainable parameters found for stage={stage}")

        # 每个 stage 都重建全新的 optimizer，避免复用上一阶段动量
        self.optimizer = AdamW(
            param_groups,
            betas=(0.9, 0.98),
            eps=1e-8,
        )

        epochs_for_scheduler = self.stage1_epochs if stage == 1 else self.epochs
        total_steps = epochs_for_scheduler * len(self.train_dataloader)
        warmup_steps = int(total_steps * 0.1)

        self.scheduler = self._get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        trainable_names = [n for n, p in self.model.engine.named_parameters() if p.requires_grad]
        print(f"[Stage2 Trainable] {trainable_names}")

    def _get_cosine_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
        # 构建“先 warmup 再 cosine 衰减”的学习率调度器
        def lr_lambda(current_step):#current_step:当前是第几步训练，int
            if current_step < num_warmup_steps:# 在 warmup 阶段
                return float(current_step) / float(max(1, num_warmup_steps)) #预热部分线性增长
            #预热过后
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) #超过预热部分cos增长
        return LambdaLR(optimizer, lr_lambda) #返回cos学习率调度器

    def train_epoch(self, epoch_idx: int, stage: int = 2):
        # 从训练数据中取出每批次对参数进行更新；记录三种损失；返回此轮次平均批损失；

        self.model.train()#切换到训练模式（启用 dropout、BN 的训练行为等）
        total_loss, total_ce, total_cl = 0.0, 0.0, 0.0
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Stage {stage} Train Epoch {epoch_idx}/{self.stage1_epochs if stage == 1 else self.epochs}",
            leave=False,
            disable=True
        )

        for step, batch_data in enumerate(progress_bar):
            history_seq = batch_data['item_id_list'].to(self.device)#[bs,seq_l]
            target_item = batch_data['item_id'].to(self.device)#[bs]
            self.optimizer.zero_grad()#上一批次梯度清零
            loss_total, loss_ce, loss_cl = self.model.calculate_loss(
                history_seq,
                target_item,
                stage=stage
            )
            loss_total.backward()
            self.optimizer.step()# 根据当前梯度更新模型参数
            self.scheduler.step()# 正常更新学习率

            total_loss += loss_total.item()
            total_ce += loss_ce.item()
            total_cl += loss_cl.item()

            current_lr = self.scheduler.get_last_lr()[0]# 读取当前学习率
            progress_bar.set_postfix({'Loss': f"{loss_total.item():.4f}", 'LR': f"{current_lr:.2e}"})# 在进度条后显示当前 batch 的 loss 和学习率

        num_batches = len(self.train_dataloader)# 当前 epoch 的 batch 总数
        # 返回该 epoch 的平均总损失、平均 CE 损失、平均 CL 损失
        return total_loss / num_batches, total_ce / num_batches, total_cl / num_batches


    def _check_codebook_health(self):
        '''
        从当前embdding取出所有物品embedding；
        根据当前量化层权重，得到量化物品的sid;
        计算平均使用率，冲突率；
        '''
        codebook_metrics = {}
        if hasattr(self.model, 'engine') and hasattr(self.model.engine, 'item_embedding'):
            self.model.eval()
            with torch.no_grad():
                all_embs = self.model.engine.item_embedding.weight# 取出全量 item embedding 权重矩阵：[num_item,embd]
                _, all_sids, _ = self.model.engine.quantizer(all_embs)#[num_item,m_layer]
                codebook_metrics = self.evaluator.evaluate_codebook_health(all_sids)
                #每层的使用率 平均使用率；冲突率；
        return codebook_metrics

    def valid_epoch(self, epoch_idx: int):
        '''
        从验证集中取出每批数据，进行推理得到预测结果；
        累积轮次所有预测结果和target_item,计算排名指标和健康度并返回；
        '''
        self.model.eval()
        if self.valid_dataloader is None:
            return 0.0, {}

        recbole_config = getattr(self.valid_dataloader, "config", None)
        if recbole_config is not None:
            metrics = evaluate_fullsort_recbole(
                self.model,
                self.valid_dataloader,
                recbole_config,
                self.device,
            )
            codebook_metrics = self._check_codebook_health()
            return float(metrics[f'ndcg@{self.val_k}']), codebook_metrics

        raise RuntimeError("Validation requires a RecBole dataloader with full-sort config.")


    def train(self):
        '''
        阶段1:调用阶段1训练 量化和embedding层；每3轮次检查是否码本崩塌；
        阶段2：调用阶段2训练transformer层；不验证，直接打印损失；
             验证：调用验证轮次返回指标；检查健康；检查剪枝；
                  根据健康和分数，决定保存模型/早停+1；
                  检查是否早停；
             返回最佳分数；

        '''
        print("=" * 50)
        print(f"开始两阶段训练在 {self.device}...")

        #阶段1：
        if self.stage1_epochs > 0:
            print("\n" + "-" * 50)
            print(f">>>阶段1：预训练RQ-VAE和物品embedding层({self.stage1_epochs} Epochs) <<<")

            self._setup_optimizer_and_scheduler(stage=1)
            # 在 stage1 开始前，先算一个“最早允许健康剪枝”的 epoch
            steps_per_epoch = len(self.train_dataloader)
            quantizer = self.model.engine.quantizer

            min_health_epoch = max(
                3,
                math.ceil(
                    max(
                        quantizer.dead_code_warmup_steps,
                        quantizer.dead_code_patience,
                    ) / max(1, steps_per_epoch)
                )
            )
            for epoch in range(1, self.stage1_epochs + 1):
                avg_train_loss, avg_train_ce, avg_train_cl = self.train_epoch(epoch, stage=1)

                self.writer.add_scalar('Stage1/Train_Total_Loss', avg_train_loss, epoch)
                log_str = f"Stage 1 - Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | CL Loss: {avg_train_cl:.4f}"

                if epoch % 3 == 0 or epoch == self.stage1_epochs:#只进行码本健康检查，不进行验证；
                    cb_metrics = self._check_codebook_health()
                    if cb_metrics:
                        act_rate = cb_metrics.get('Active_Rate_Mean', 1.0)
                        coll_rate = cb_metrics.get('Collision_Rate', 0.0)
                        log_str += f" | ActRate: {act_rate:.2f} | CollRate: {coll_rate:.4f}"
                        if epoch < min_health_epoch:
                            log_str += f" | HealthCheckOnly (prune after epoch {min_health_epoch})"
                        elif act_rate < 0.02 and coll_rate > 0.95:
                            log_str += " ⚠️[阶段1：出现码本崩塌]"
                            print(log_str)
                            if self.trial is not None:
                                raise optuna.exceptions.TrialPruned()
                            else:
                                print("遇到码本崩塌，提前终止 Stage 1 训练。")
                                break  # 加入普通的 break 防止无效训练
                print(log_str)

        #阶段2：
        print("\n" + "-" * 50)
        print(f">>> 阶段2：训练transformer预测层 ({self.epochs} Epochs) <<<")

        self._setup_optimizer_and_scheduler(stage=2)

        best_valid_score = -float('inf')#最佳验证分数
        patience_counter = 0# 初始化早停计数器
        warmup_epochs = int(self.epochs * 0.2)#预热轮次：不验证
        val_freq_early = 3# 前期每5个epoch 验证一次
        val_freq_late = 1# 后期每个epoch都验证
        mid_point = int(self.epochs * 0.6)# 前后期的分界点：前 60%为前期，后40%为后期

        for epoch in range(1, self.epochs + 1):
            avg_train_loss, avg_train_ce, avg_train_cl = self.train_epoch(epoch, stage=2)
            self.writer.add_scalar('Stage2/Train_Total_Loss', avg_train_loss, epoch)
            self.writer.add_scalar('Stage2/Train_CE_Loss', avg_train_ce, epoch)
            self.writer.add_scalar('Stage2/Train_CL_Loss', avg_train_cl, epoch)

            if hasattr(self.model.engine, 'log_vars'):
                with torch.no_grad():
                    weight_ce = torch.exp(-self.model.engine.log_vars[0]).item()
                    weight_cl = torch.exp(-self.model.engine.log_vars[1]).item()
                    weight_commit = torch.exp(-self.model.engine.log_vars[2]).item()

                    self.writer.add_scalar('Stage2_Weights/CE_Weight', weight_ce, epoch)
                    self.writer.add_scalar('Stage2_Weights/CL_Weight', weight_cl, epoch)
                    self.writer.add_scalar('Stage2_Weights/Commit_Weight', weight_commit, epoch)

            log_str = f"Stage 2 - Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f}"

            is_val_epoch = False
            if self.valid_dataloader is not None:
                if epoch > warmup_epochs:
                    if epoch <= mid_point and epoch % val_freq_early == 0:#前期验证
                        is_val_epoch = True
                    elif epoch > mid_point and epoch % val_freq_late == 0:
                        is_val_epoch = True

                if epoch == self.epochs:#最后一个轮次必须验证
                    is_val_epoch = True

            if is_val_epoch:
                valid_ndcg, codebook_metrics = self.valid_epoch(epoch)
                self.writer.add_scalar(f'Stage2/Valid_NDCG_{self.val_k}', valid_ndcg, epoch)
                log_str += f" | Valid NDCG@{self.val_k}: {valid_ndcg:.4f}"

                is_healthy = True
                if codebook_metrics:
                    act_rate = codebook_metrics['Active_Rate_Mean']
                    coll_rate = codebook_metrics['Collision_Rate']

                    self.writer.add_scalar('Stage2_Codebook/Active_Rate_Mean', act_rate, epoch)
                    self.writer.add_scalar('Stage2_Codebook/Collision_Rate', coll_rate, epoch)

                    log_str += f" | ActRate: {act_rate:.2f} | CollRate: {coll_rate:.4f}"

                    if act_rate < 0.03 or coll_rate > 0.95:
                        is_healthy = False
                        log_str += " [Unhealthy]"

                #Optuna 剪枝逻辑 (只在验证轮次触发)
                if self.trial is not None:
                    self.trial.report(valid_ndcg, epoch)
                    if self.trial.should_prune():
                        self.writer.close()
                        print(log_str)
                        print(f"\n[Pruning] ⚠️ Trial {self.trial.number} pruned at epoch {epoch}.")
                        raise optuna.exceptions.TrialPruned()#抛出异常，中止当前 trial

                improved = valid_ndcg > best_valid_score
                #早停与最佳模型保存:
                if improved:
                    best_valid_score = valid_ndcg
                    patience_counter = 0#早停计数器清0
                    torch.save(
                        {
                            "state_dict": self.model.state_dict(),
                            "item_loss_weight": float(
                                getattr(self.model.engine, "item_loss_weight", 0.0)
                            ),
                        },
                        self.best_model_path,
                    )
                    log_str += f" --> [Best Model Saved!]"
                    if not is_healthy:
                        log_str += " [Warning: codebook unhealthy]"
                else:
                    # 如果当前分数没有提升，或者 codebook 不健康
                    patience_counter += 1
                    reason = "Score drop"
                    if not is_healthy:
                        reason += " + codebook warning"
                    log_str += f" | Patience: {patience_counter}/{self.patience} ({reason})"

                print(log_str)

                if patience_counter >= self.patience:
                    #已经达到最大容忍次数
                    print(f"\n[阶段2：在第 {epoch}] 早停，训练停止！")
                    break
            else:
                # 非验证轮次，直接打印训练日志
                print(log_str)

        self.writer.close()
        print(f"\n[Trainer] 两阶段训练结束.")

        return best_valid_score# 返回训练过程中获得的最佳验证分数
