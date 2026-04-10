import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from torch.utils.tensorboard import SummaryWriter
from config import args
from dataloader import get_data_loader
from utils import remove_dir_and_create_dir, create_model, set_seed
from tools import EarlyStopping


def try_get_block_data(args, space_label, time_label):
    """
    尝试加载某个 (space_label, time_label) 子块的数据。
    若该组合无效、为空或加载失败，则返回 None。
    """
    try:
        data = get_data_loader(
            args.dataset_dir,
            args.batch_size,
            args.num_workers,
            time_label=time_label,
            space_label=space_label,
            train_ratio=0.8
        )
    except Exception as e:
        print(f" 子块 S{space_label}_T{time_label} 加载失败: {e}")
        return None

    if data is None:
        print(f" 子块 S{space_label}_T{time_label} 返回 None，跳过")
        return None

    try:
        (
            train_loader,
            val_loader,
            train_dataset,
            val_dataset,
            output_channel,
            d_indices,
            time_idx,
            val_time_idx
        ) = data
    except Exception as e:
        print(f" 子块 S{space_label}_T{time_label} 返回内容异常: {e}")
        return None

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print(f" 子块 S{space_label}_T{time_label} 数据为空，跳过")
        return None

    return data


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    weights_dir = os.path.join(args.summary_dir, "weights")
    log_dir = os.path.join(args.summary_dir, "logs")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    remove_dir_and_create_dir(weights_dir)
    remove_dir_and_create_dir(log_dir)

    writer = SummaryWriter(log_dir)
    set_seed(42)


    # 范围上限
    candidate_space_labels = range(1, 6)
    candidate_time_labels = range(0, 4)

    valid_block_count = 0

    for space_label in candidate_space_labels:
        for time_label in candidate_time_labels:
            block_name = f"S{space_label}_T{time_label}"
            print(f"尝试训练子块: {block_name}")

            block_data = try_get_block_data(args, space_label, time_label)
            if block_data is None:
                continue

            (
                train_loader,
                val_loader,
                train_dataset,
                val_dataset,
                output_channel,
                d_indices,
                time_idx,
                val_time_idx
            ) = block_data

            valid_block_count += 1
            print(f"   有效子块: {block_name}")
            print(f"   Train samples: {len(train_dataset)}")
            print(f"   Val samples:   {len(val_dataset)}")
            print(f"   Output depth:  {output_channel}")

            model = create_model(output_c=output_channel).to(device)

            loss_function = torch.nn.MSELoss()
            optimizer = optim.SGD(
                model.parameters(),
                lr=args.lr,
                momentum=0.9,
                weight_decay=5e-5
            )

            lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

            early_stopping = EarlyStopping(
                patience=40,
                verbose=True,
                delta=1e-4
            )

            min_val_loss = np.inf
            best_model_path = None

            # 记录“最佳 epoch”对应的验证结果
            best_all_layer_mse = None
            best_mean_train_loss = None
            best_mean_val_loss = None
            best_preds = None
            best_labels = None

            for epoch in range(args.epochs):
                # 训练阶段
                model.train()
                train_loss = []

                train_bar = tqdm(train_loader, desc=f"{block_name} Epoch {epoch}")
                for images, labels in train_bar:
                    images = images.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    logits = model(images)
                    loss = loss_function(logits, labels)
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    train_loss.append(loss.item())
                    train_bar.set_postfix(loss=f"{loss.item():.6f}")

                # 这里按 epoch 更新学习率
                scheduler.step()

                # 验证阶段
                model.eval()
                val_loss = []
                all_layer_mse = np.zeros(output_channel, dtype=np.float64)
                all_preds = []
                all_labels = []

                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(device)
                        labels = labels.to(device)

                        logits = model(images)

                        batch_val_loss = F.mse_loss(logits, labels, reduction='mean').item()
                        val_loss.append(batch_val_loss)

                        for c in range(output_channel):
                            mse = F.mse_loss(logits[:, c], labels[:, c], reduction='mean').item()
                            all_layer_mse[c] += mse * images.size(0)

                        all_preds.append(logits.cpu().numpy())
                        all_labels.append(labels.cpu().numpy())

                total_val_samples = len(val_dataset)
                all_layer_mse /= total_val_samples

                mean_train_loss = float(np.mean(train_loss)) if len(train_loss) > 0 else np.nan
                mean_val_loss = float(np.mean(val_loss)) if len(val_loss) > 0 else np.nan

                print(
                    f"{block_name} | Epoch {epoch:03d} | "
                    f"Train Loss: {mean_train_loss:.6f} | Val Loss: {mean_val_loss:.6f}"
                )

                writer.add_scalar(f"{block_name}/Train_Loss", mean_train_loss, epoch)
                writer.add_scalar(f"{block_name}/Val_Loss", mean_val_loss, epoch)
                writer.add_scalar(f"{block_name}/LR", optimizer.param_groups[0]['lr'], epoch)


                # 保存最佳模型
                if mean_val_loss < min_val_loss:
                    min_val_loss = mean_val_loss

                    best_model_path = os.path.join(
                        weights_dir,
                        f"{block_name}_epoch={epoch}_val_loss={min_val_loss:.4f}.pth"
                    )
                    torch.save(model.state_dict(), best_model_path)

                    best_all_layer_mse = all_layer_mse.copy()
                    best_mean_train_loss = mean_train_loss
                    best_mean_val_loss = mean_val_loss
                    best_preds = np.concatenate(all_preds, axis=0)
                    best_labels = np.concatenate(all_labels, axis=0)

                    print(f" 保存最佳模型: {best_model_path}")

                # Early stopping
                early_stopping(mean_val_loss)
                if early_stopping.early_stop:
                    print(f" {block_name} 提前停止")
                    break

            # 若整个 block 没有成功训练出结果，则跳过保存
            if best_all_layer_mse is None:
                print(f" {block_name} 未产生有效结果，跳过保存")
                continue

            # 保存每层验证 MSE（最佳模型对应）
            depthwise_csv = os.path.join(weights_dir, "depthwise_val_mse.csv")
            pd.DataFrame([
                {
                    "space_label": space_label,
                    "time_label": time_label,
                    "depth_layer": i,
                    "val_mse": best_all_layer_mse[i]
                }
                for i in range(output_channel)
            ]).to_csv(
                depthwise_csv,
                mode="a",
                header=not os.path.exists(depthwise_csv),
                index=False
            )

            # 保存每个子块的 summary loss
            summary_csv = os.path.join(weights_dir, "summary_loss.csv")
            pd.DataFrame([{
                "space_label": space_label,
                "time_label": time_label,
                "train_loss": best_mean_train_loss,
                "val_loss": best_mean_val_loss,
                "best_model_path": os.path.basename(best_model_path) if best_model_path else ""
            }]).to_csv(
                summary_csv,
                mode="a",
                header=not os.path.exists(summary_csv),
                index=False
            )

            # 保存最佳模型对应的预测结果与标签
            np.save(
                os.path.join(weights_dir, f"{block_name}_preds.npy"),
                best_preds
            )
            np.save(
                os.path.join(weights_dir, f"{block_name}_labels.npy"),
                best_labels
            )
            np.save(
                os.path.join(weights_dir, f"{block_name}_d_indices.npy"),
                d_indices
            )
            np.save(
                os.path.join(weights_dir, f"{block_name}_time_indices.npy"),
                val_time_idx
            )
            np.save(
                os.path.join(weights_dir, f"{block_name}_depth_len.npy"),
                output_channel
            )

            print(f" {block_name} 全部结果已保存")

    writer.close()
    print(f"\n 训练结束，共找到有效子块数: {valid_block_count}")


if __name__ == "__main__":
    main(args)