import os
import re
import glob
import numpy as np
import pandas as pd
import joblib



# 1. 路径配置
weights_dir = "1206_cluster/weights"         
scaler_path = "scalers/target_scaler.pkl"     


# 2. 全局尺寸配置
H, W = 21, 21        
D_total = 73         

# 是否保存合并后的全局数组
save_merged_arrays = True


# 3. 自动扫描所有有效子块
pred_files = glob.glob(os.path.join(weights_dir, "S*_T*_preds.npy"))

pattern = re.compile(r"S(\d+)_T(\d+)_preds\.npy$")

block_list = []
for pred_path in pred_files:
    filename = os.path.basename(pred_path)
    match = pattern.match(filename)
    if match is None:
        continue

    space_label = int(match.group(1))
    time_label = int(match.group(2))
    block_name = f"S{space_label}_T{time_label}"

    label_path = os.path.join(weights_dir, f"{block_name}_labels.npy")
    time_idx_path = os.path.join(weights_dir, f"{block_name}_time_indices.npy")
    d_idx_path = os.path.join(weights_dir, f"{block_name}_d_indices.npy")

    if all(os.path.exists(p) for p in [pred_path, label_path, time_idx_path, d_idx_path]):
        block_list.append({
            "block_name": block_name,
            "space_label": space_label,
            "time_label": time_label,
            "pred_path": pred_path,
            "label_path": label_path,
            "time_idx_path": time_idx_path,
            "d_idx_path": d_idx_path,
        })

if len(block_list) == 0:
    raise FileNotFoundError(
        f"在 {weights_dir} 下没有找到符合格式的子块文件，如 S*_T*_preds.npy"
    )

print(f" 检测到有效子块数: {len(block_list)}")
for item in block_list:
    print(f"   - {item['block_name']}")


# 4. 自动推断全局时间长度 T_total
T_total = 0
for item in block_list:
    time_idx = np.load(item["time_idx_path"])
    if len(time_idx) == 0:
        continue
    T_total = max(T_total, int(np.max(time_idx)) + 1)

if T_total == 0:
    raise ValueError("未能从 time_indices 中推断出有效的全局时间长度 T_total")

print(f"\n 检测到全局时间长度 T_total = {T_total}")


# 5. 初始化全局预测与标签
global_preds = np.full((T_total, D_total, H, W), np.nan, dtype=np.float32)
global_labels = np.full((T_total, D_total, H, W), np.nan, dtype=np.float32)

# 记录每个位置是否被覆盖过，方便检查冲突
fill_count = np.zeros((T_total, D_total), dtype=np.int32)


# 6. 合并所有子块结果
print("\n 开始合并所有子块预测结果...")

for item in block_list:
    block_name = item["block_name"]

    try:
        preds = np.load(item["pred_path"])         # (N, D_sub, H, W)
        labels = np.load(item["label_path"])       # (N, D_sub, H, W)
        time_idx = np.load(item["time_idx_path"]) # (N,)
        d_indices = np.load(item["d_idx_path"])   # (D_sub,)

        # ---------- 基本检查 ----------
        if preds.ndim != 4 or labels.ndim != 4:
            print(f" {block_name} 的 preds/labels 维度不是 4 维，跳过")
            continue

        if preds.shape != labels.shape:
            print(f" {block_name} 的 preds 和 labels 形状不一致，跳过")
            print(f"   preds.shape  = {preds.shape}")
            print(f"   labels.shape = {labels.shape}")
            continue

        N, D_sub, H_sub, W_sub = preds.shape

        if H_sub != H or W_sub != W:
            print(f" {block_name} 的空间尺寸不匹配，跳过")
            print(f"   期望: ({H}, {W}), 实际: ({H_sub}, {W_sub})")
            continue

        if len(time_idx) != N:
            print(f" {block_name} 中 time_idx 长度与样本数不一致，跳过")
            print(f"   len(time_idx) = {len(time_idx)}, N = {N}")
            continue

        if len(d_indices) != D_sub:
            print(f" {block_name} 中 d_indices 长度与深度数不一致，跳过")
            print(f"   len(d_indices) = {len(d_indices)}, D_sub = {D_sub}")
            continue

        # 转成 int，防止索引出问题
        time_idx = time_idx.astype(int)
        d_indices = d_indices.astype(int)

        # 索引合法性检查
        if np.any(time_idx < 0) or np.any(time_idx >= T_total):
            print(f" {block_name} 中存在越界的 time_idx，跳过")
            continue

        if np.any(d_indices < 0) or np.any(d_indices >= D_total):
            print(f" {block_name} 中存在越界的 d_indices，跳过")
            continue

        # ---------- 写入全局数组 ----------
        for i, t in enumerate(time_idx):
            # 若同一个 (t, d_indices) 被重复写入，可以打印提醒
            if np.any(fill_count[t, d_indices] > 0):
                print(f" {block_name} 在时间 t={t} 的部分深度层存在重复覆盖")

            global_preds[t, d_indices, :, :] = preds[i]
            global_labels[t, d_indices, :, :] = labels[i]
            fill_count[t, d_indices] += 1

        print(f" 合并完成: {block_name} | 样本数={N} | 深度数={D_sub}")

    except Exception as e:
        print(f" 合并 {block_name} 出错: {e}")


# 7. 可选：保存合并后的归一化状态数组
if save_merged_arrays:
    np.save(os.path.join(weights_dir, "global_preds_merged_before_inverse.npy"), global_preds)
    np.save(os.path.join(weights_dir, "global_labels_merged_before_inverse.npy"), global_labels)
    print("\n 已保存合并后的归一化数组")


# 8. 逆归一化
if os.path.exists(scaler_path):
    print("\n🌡️ 检测到 scaler，开始执行逆归一化...")
    scaler = joblib.load(scaler_path)

    T, D, H_, W_ = global_preds.shape
    preds_flat = global_preds.reshape(T, -1)
    labels_flat = global_labels.reshape(T, -1)

    preds_inv_flat = preds_flat.copy()
    labels_inv_flat = labels_flat.copy()

    # 只对“整条时间样本都没有 NaN”的行做 inverse_transform
    valid_row_mask = (~np.isnan(preds_flat).any(axis=1)) & (~np.isnan(labels_flat).any(axis=1))

    valid_count = int(valid_row_mask.sum())
    print(f"   可逆归一化的完整时间样本数: {valid_count}/{T}")

    if valid_count > 0:
        preds_inv_flat[valid_row_mask] = scaler.inverse_transform(preds_flat[valid_row_mask])
        labels_inv_flat[valid_row_mask] = scaler.inverse_transform(labels_flat[valid_row_mask])

        global_preds = preds_inv_flat.reshape(T, D, H_, W_)
        global_labels = labels_inv_flat.reshape(T, D, H_, W_)

        print(" 逆归一化完成")
    else:
        print(" 没有完整时间样本可用于逆归一化，跳过 inverse_transform")
else:
    print("\n 未找到 scaler 文件，跳过逆归一化")


# 9. 计算每层 RMSE / MAE

print("\n 开始计算每层 RMSE / MAE...")

rmse_per_layer = []
mae_per_layer = []
valid_points_per_layer = []

for d in range(D_total):
    pred_d = global_preds[:, d, :, :].reshape(-1)
    label_d = global_labels[:, d, :, :].reshape(-1)

    mask = (~np.isnan(pred_d)) & (~np.isnan(label_d))
    valid_num = int(mask.sum())
    valid_points_per_layer.append(valid_num)

    if valid_num > 0:
        diff = label_d[mask] - pred_d[mask]
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        mae = float(np.mean(np.abs(diff)))
    else:
        rmse = np.nan
        mae = np.nan

    rmse_per_layer.append(rmse)
    mae_per_layer.append(mae)


# 10. 计算整体 Overall RMSE / MAE

pred_all = global_preds.reshape(-1)
label_all = global_labels.reshape(-1)
mask_all = (~np.isnan(pred_all)) & (~np.isnan(label_all))

if mask_all.sum() > 0:
    diff_all = label_all[mask_all] - pred_all[mask_all]
    rmse_all = float(np.sqrt(np.mean(diff_all ** 2)))
    mae_all = float(np.mean(np.abs(diff_all)))
    print(f"\n📈 Overall Metrics")
    print(f"   RMSE = {rmse_all:.6f}")
    print(f"   MAE  = {mae_all:.6f}")
    print(f"   有效点数 = {int(mask_all.sum())}")
else:
    rmse_all = np.nan
    mae_all = np.nan
    print("\n 没有有效点可用于计算 Overall RMSE / MAE")


# 11. 保存每层指标
depth_metrics_df = pd.DataFrame({
    "depth_layer": np.arange(D_total),
    "RMSE": rmse_per_layer,
    "MAE": mae_per_layer,
    "valid_points": valid_points_per_layer
})

depth_metrics_csv = os.path.join(weights_dir, "global_rmse_mae_per_depth.csv")
depth_metrics_df.to_csv(depth_metrics_csv, index=False)

print(f"\n 每层 RMSE / MAE 已保存到: {depth_metrics_csv}")


# 12. 保存整体指标
overall_metrics_df = pd.DataFrame([{
    "overall_RMSE": rmse_all,
    "overall_MAE": mae_all,
    "valid_points": int(mask_all.sum())
}])

overall_metrics_csv = os.path.join(weights_dir, "global_overall_metrics.csv")
overall_metrics_df.to_csv(overall_metrics_csv, index=False)

print(f" Overall 指标已保存到: {overall_metrics_csv}")

# 13. 保存逆归一化后的全局数组
if save_merged_arrays:
    np.save(os.path.join(weights_dir, "global_preds_merged.npy"), global_preds)
    np.save(os.path.join(weights_dir, "global_labels_merged.npy"), global_labels)
    print("已保存逆归一化后的全局预测与标签数组")

print("\n merge_and_evaluate 全部完成！")