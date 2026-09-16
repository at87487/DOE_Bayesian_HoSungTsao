import numpy as np
import pandas as pd
import warnings
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import gradio as gr
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore")

def setup_matplotlib_font():
    system_fonts = [f.name for f in fm.fontManager.ttflist]
    candidate_fonts = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'PingFang SC', 'Heiti TC', 'Noto Sans CJK JP', 'DejaVu Sans']
    chosen_font = next((f for f in candidate_fonts if f in system_fonts), 'DejaVu Sans')
    plt.rcParams['font.sans-serif'] = [chosen_font, 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

setup_matplotlib_font()


def parse_factors_str(factors_str):
    cont_factors, discrete_factors = {}, {}
    if not isinstance(factors_str, str):
        return cont_factors, discrete_factors

    for item in factors_str.split(";"):
        if ":" in item:
            try:
                parts = item.split(":")
                name = parts[0].strip()
                vals_part = parts[1].strip()

                is_int = "int" in vals_part.lower() or "discrete" in vals_part.lower()
                clean_vals = vals_part.replace("(int)", "").replace("(discrete)", "").replace("(cont)", "")
                levels = [float(x.strip()) for x in clean_vals.split(",") if x.strip()]

                if len(levels) >= 2:
                    if is_int:
                        discrete_factors[name] = (int(min(levels)), int(max(levels)))
                    else:
                        cont_factors[name] = (min(levels), max(levels))
                elif len(levels) == 1:
                    if is_int:
                        discrete_factors[name] = (int(levels[0]), int(levels[0]))
                    else:
                        cont_factors[name] = (levels[0], levels[0])
            except Exception:
                continue
    return cont_factors, discrete_factors


def generate_engineering_doe_matrix(factors_str, target_runs=20, enable_replication=False):
    np.random.seed(42)
    cont_factors, discrete_factors = parse_factors_str(factors_str)
    all_factors = {**cont_factors, **discrete_factors}
    k = len(all_factors)
    var_names = list(all_factors.keys())
    if k == 0:
        return pd.DataFrame(), 0

    actual_runs = max(target_runs, k + 2)
    sampler = qmc.LatinHypercube(d=k, seed=42)
    sample_unit = sampler.random(n=actual_runs)

    l_bounds = np.array([all_factors[v][0] for v in var_names])
    u_bounds = np.array([all_factors[v][1] for v in var_names])

    for i in range(len(u_bounds)):
        if u_bounds[i] <= l_bounds[i]:
            u_bounds[i] = l_bounds[i] + 1.0

    sample_scaled = qmc.scale(sample_unit, l_bounds, u_bounds)
    df_base = pd.DataFrame(sample_scaled, columns=var_names)

    for name in discrete_factors:
        if name in df_base.columns:
            df_base[name] = np.round(df_base[name]).astype(int)

    expanded_rows = []
    for idx, row in df_base.iterrows():
        if enable_replication:
            reps = int(2 + (idx % 3) if k <= 3 else 2 + ((idx + k) % 3))
        else:
            reps = 1

        for r in range(reps):
            row_copy = row.copy()
            row_copy["Run_ID"] = f"EXP-{idx + 1:03d}_R{r + 1}"
            row_copy["Group_ID"] = f"EXP-{idx + 1:03d}"
            row_copy["Replicate_Index"] = r + 1
            expanded_rows.append(row_copy)

    df_cand = pd.DataFrame(expanded_rows)
    cols = ["Run_ID", "Group_ID", "Replicate_Index"] + [c for c in df_cand.columns if c not in ["Run_ID", "Group_ID", "Replicate_Index"]]
    df_cand = df_cand[cols]
    return df_cand, k


def parse_specs_and_targets_str(specs_str):
    specs, targets = {}, {}
    if not isinstance(specs_str, str):
        return {"Depth(µm)": "continuous"}, {"Depth(µm)": 100.0}

    for item in specs_str.split(";"):
        if ":" in item:
            try:
                parts = [p.strip() for p in item.split(":")]
                name = parts[0]
                if not name: continue
                stype, target_val = ("binary" if len(parts) >= 2 and "binary" in parts[1].lower() else "continuous"), 100.0
                if len(parts) >= 3:
                    try: target_val = float(parts[2])
                    except ValueError: pass
                specs[name], targets[name] = stype, target_val
            except Exception:
                continue

    if not specs:
        specs, targets = {"Depth(µm)": "continuous", "Angle(°)": "continuous", "Damage(0/1)": "binary"}, {"Depth(µm)": 100.0, "Angle(°)": 45.0, "Damage(0/1)": 0.0}
    return specs, targets


def update_dynamic_specs_ui(specs_str):
    specs_dict, targets_dict = parse_specs_and_targets_str(specs_str)
    cont_specs = [name for name, stype in specs_dict.items() if stype == "continuous"]
    binary_specs = [name for name, stype in specs_dict.items() if stype == "binary"]

    updates = []
    for i in range(3):
        if i < len(cont_specs):
            s_name = cont_specs[i]
            updates.extend([
                gr.update(visible=True, label=f"連續規格 [{s_name}] 策略", value="目標與容忍區間"),
                gr.update(visible=True, label=f"[{s_name}] 目標值", value=targets_dict.get(s_name, 100.0)),
                gr.update(visible=True, label="下限容許百分比 (%)", value=0.0),
                gr.update(visible=True, label="上限容許百分比 (%)", value=10.0)
            ])
        else:
            updates.extend([gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)])

    for i in range(3):
        updates.append(gr.update(visible=True, label=f"二元分類護欄：{s_name if i < len(binary_specs) else ''} (上限機率)", value=0.0) if i < len(binary_specs) else gr.update(visible=False))

    return tuple(updates)


def generate_stage1_matrix(factors_str, specs_str, assign_mode, enable_replication):
    try:
        cont_factors, discrete_factors = parse_factors_str(factors_str)
        k = len(cont_factors) + len(discrete_factors)
        specs_dict, _ = parse_specs_and_targets_str(specs_str)
        if k < 1:
            return None, *([gr.update(visible=False)] * 15), "錯誤：請至少設定 1 個有效的控制因子。"

        target_r = max(15, k * 3) if "空間填充" in assign_mode else max(8, k + 2)
        df_final, _ = generate_engineering_doe_matrix(factors_str, target_runs=target_r, enable_replication=enable_replication)

        for spec_name, spec_type in specs_dict.items():
            df_final[spec_name] = 0.0 if spec_type == "continuous" else 0

        rep_status = "已啟動統計學客觀重複次數指派 (多筆展開)" if enable_replication else "單次實驗模式"
        info_msg = f"實驗矩陣建立完成。模式：{assign_mode} | {rep_status} | 總資料列數：{len(df_final)} 筆"
        return df_final, *update_dynamic_specs_ui(specs_str), info_msg

    except Exception as e:
        return None, *([gr.update(visible=False)] * 15), f"矩陣生成失敗: {str(e)}"


def simulate_data_fill_with_targets(df, specs_str):
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return None, "請先生成實驗矩陣。"
    try:
        np.random.seed(int(time.time()) % 1000)
        df_sim = df.copy()
        specs_dict, targets_dict = parse_specs_and_targets_str(specs_str)
        meta_cols = ["Run_ID", "Group_ID", "Replicate_Index"]
        x_cols = [c for c in df_sim.columns if c not in meta_cols and c not in specs_dict]

        for c in x_cols:
            df_sim[c] = pd.to_numeric(df_sim[c], errors='coerce').fillna(0.0)

        x_means = {c: df_sim[c].mean() for c in x_cols}
        x_stds = {c: (df_sim[c].std() if df_sim[c].std() > 0 else 1.0) for c in x_cols}

        sim_rows = []
        for _, row in df_sim.iterrows():
            norm_effect = sum([(row[c] - x_means[c]) / x_stds[c] for c in x_cols if x_stds[c] > 0])
            rep_idx = row.get("Replicate_Index", 1)
            for spec_name, spec_type in specs_dict.items():
                if spec_type == "binary":
                    row[spec_name] = 1 if np.random.rand() < (0.1 + 0.02 * rep_idx) else 0
                else:
                    noise = np.random.normal(0, 1.2 + 0.3 * rep_idx)
                    row[spec_name] = round(float(targets_dict.get(spec_name, 100.0) + (norm_effect * 1.2) + noise), 2)
            sim_rows.append(row)

        return pd.DataFrame(sim_rows), "測試數據模擬完成（各重複次數獨立填寫）。"
    except Exception as e:
        return df, f"模擬填寫失敗: {str(e)}"


def upload_real_plant_data(file_obj, current_df):
    if file_obj is None:
        return current_df, "請選擇有效的 CSV 檔案。"
    try:
        df_uploaded = pd.read_csv(file_obj.name)
        if "Group_ID" not in df_uploaded.columns:
            df_uploaded["Group_ID"] = df_uploaded["Run_ID"]
        if "Replicate_Index" not in df_uploaded.columns:
            df_uploaded["Replicate_Index"] = 1
        return df_uploaded, f"成功載入數據，共 {len(df_uploaded)} 筆紀錄。"
    except Exception as e:
        return current_df, f"檔案解析失敗: {str(e)}"


def feed_back_suggestions_to_matrix(current_df, suggestion_df, specs_str):
    if suggestion_df is None or (isinstance(suggestion_df, pd.DataFrame) and suggestion_df.empty):
        return current_df, "目前已達收斂狀態，無新增補點。"
    if current_df is None or (isinstance(current_df, pd.DataFrame) and current_df.empty):
        return suggestion_df, "已設定為新的實驗矩陣。"

    try:
        df_curr = current_df.copy()
        df_sugg = suggestion_df.copy()
        specs_dict, _ = parse_specs_and_targets_str(specs_str)

        for s_name, s_type in specs_dict.items():
            if s_name not in df_sugg.columns:
                df_sugg[s_name] = 0.0 if s_type == "continuous" else 0

        if "Group_ID" not in df_sugg.columns:
            df_sugg["Group_ID"] = df_sugg["Run_ID"]
        if "Replicate_Index" not in df_sugg.columns:
            df_sugg["Replicate_Index"] = 1

        start_idx = len(df_curr['Group_ID'].unique()) + 1 if 'Group_ID' in df_curr.columns else len(df_curr) + 1
        df_sugg["Run_ID"] = [f"ITER-EXP-{i+start_idx:03d}_R1" for i in range(len(df_sugg))]
        df_sugg["Group_ID"] = [f"ITER-EXP-{i+start_idx:03d}" for i in range(len(df_sugg))]

        df_combined = pd.concat([df_curr, df_sugg], ignore_index=True)
        return df_combined, f"已成功追加 {len(df_sugg)} 筆推薦點至實驗矩陣，總計 {len(df_combined)} 筆。"
    except Exception as e:
        return current_df, f"參數導回失敗: {str(e)}"


def auto_generate_gpr_robust_sequential_doe(current_X_df, control_features, gp_models, requested_batch_size, enable_replication=True):
    np.random.seed(int(time.time()) % 1000)
    k = len(control_features)
    bounds = [(current_X_df[col].min(), current_X_df[col].max()) for col in control_features]

    sampler = qmc.LatinHypercube(d=k, seed=int(time.time()) % 10000)
    cand_unit = sampler.random(n=500)
    l_bounds = np.array([b[0] for b in bounds])
    u_bounds = np.array([b[1] for b in bounds])
    X_cand_arr = qmc.scale(cand_unit, l_bounds, u_bounds)

    df_cand = pd.DataFrame(X_cand_arr, columns=control_features)

    uncertainties, means = [], []
    for name, model_info in gp_models.items():
        if model_info["type"] == "continuous":
            pred_m, stds = model_info["model"].predict(X_cand_arr, return_std=True)
            means.append(pred_m)
            uncertainties.append(stds)

    if uncertainties and means:
        mean_std = np.mean(uncertainties, axis=0)
        mean_pred = np.mean(means, axis=0)
        df_cand["Robust_Score"] = mean_pred + 1.5 * mean_std
    else:
        df_cand["Robust_Score"] = np.random.rand(len(df_cand))

    actual_take = max(2, int(requested_batch_size))
    df_selected = df_cand.sort_values(by="Robust_Score", ascending=False).head(actual_take)

    # 擴展重複實驗展開邏輯
    expanded_rows = []
    for idx, row in df_selected.iterrows():
        reps = int(2 + (idx % 3) if k <= 3 else 2 + ((idx + k) % 3)) if enable_replication else 1
        for r in range(reps):
            row_copy = row.copy()
            row_copy["Run_ID"] = f"SEQ-{idx + 1:03d}_R{r + 1}"
            row_copy["Group_ID"] = f"SEQ-{idx + 1:03d}"
            row_copy["Replicate_Index"] = r + 1
            expanded_rows.append(row_copy)

    final_suggestion = pd.DataFrame(expanded_rows)
    cols = ["Run_ID", "Group_ID", "Replicate_Index"] + [c for c in final_suggestion.columns if c not in ["Run_ID", "Group_ID", "Replicate_Index"]]
    final_suggestion = final_suggestion[cols].reset_index(drop=True)

    reasoning = f"基於模型不確定性推薦 {actual_take} 個候選群組，並已自動展開重複測值。"
    return final_suggestion, actual_take, reasoning


def analyze_multi_objective_doe_robust(df, specs_str, c1_m, c1_t, c1_min, c1_max, c2_m, c2_t, c2_min, c2_max, c3_m, c3_t, c3_min, c3_max, bin1, bin2, bin3, robust_weight, batch_size_slider):
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return "請先準備實驗數據。", pd.DataFrame()

    try:
        df_data = df.copy()
        specs_dict, targets_dict = parse_specs_and_targets_str(specs_str)

        cont_settings = [(c1_m, c1_t, c1_min, c1_max), (c2_m, c2_t, c2_min, c2_max), (c3_m, c3_t, c3_min, c3_max)]
        cont_specs = [n for n, t in specs_dict.items() if t == "continuous"]
        binary_names = [n for n, t in specs_dict.items() if t == "binary"]

        targets_state = {s_name: {"mode": cont_settings[idx][0], "target": float(cont_settings[idx][1] or targets_dict.get(s_name, 100.0)), "min_pct": float(cont_settings[idx][2] or 0.0), "max_pct": float(cont_settings[idx][3] or 10.0)} for idx, s_name in enumerate(cont_specs) if idx < len(cont_settings)}
        binary_caps = {b_name: float([bin1, bin2, bin3][idx] or 0.0) for idx, b_name in enumerate(binary_names) if idx < 3}

        y_cols = list(specs_dict.keys())
        meta_cols = ["Run_ID", "Group_ID", "Replicate_Index"]
        x_cols = [c for c in df_data.columns if c not in meta_cols and c not in y_cols]

        for c in x_cols + y_cols:
            df_data[c] = pd.to_numeric(df_data[c], errors='coerce').fillna(0.0)

        # 針對模型訓練：連續規格取平均 (mean)，二元分類規格取多數決/四捨五入 (int) 確保型態為整數 0 或 1
        if "Group_ID" in df_data.columns:
            agg_dict = {col: 'first' for col in x_cols}
            for s_name, s_type in specs_dict.items():
                if s_type == "binary":
                    agg_dict[s_name] = lambda x: int(np.round(x.mean()))
                else:
                    agg_dict[s_name] = 'mean'
            df_train = df_data.groupby("Group_ID").agg(agg_dict).reset_index()
        else:
            df_train = df_data.copy()

        X = df_train[x_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        rep = ["### 模型分析與強健性報告\n---"]
        models = {}
        for spec_name, spec_type in specs_dict.items():
            y_vals = df_train[spec_name].values
            if spec_type == "binary":
                unique_classes = np.unique(y_vals)
                if len(unique_classes) < 2:
                    class_val = int(unique_classes[0])
                    models[spec_name] = {"type": "binary_constant", "constant_val": class_val}
                    rep.append(f"- **{spec_name} (二元分類)**：資料僅含單一類別 ({class_val})，已啟用常數防護。")
                else:
                    scaler_bin = StandardScaler()
                    clf = LogisticRegression(max_iter=5000, random_state=42).fit(scaler_bin.fit_transform(X), y_vals)
                    models[spec_name] = {"type": "binary", "model": make_pipeline(scaler_bin, clf)}
                    rep.append(f"- **{spec_name} (二元分類)**：分類模型訓練完成。")
            else:
                kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e2), nu=2.5)
                gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-4, normalize_y=True, n_restarts_optimizer=5, random_state=42)
                gpr.fit(X_scaled, y_vals)
                models[spec_name] = {"type": "continuous", "scaler": scaler, "model": gpr}
                rep.append(f"- **{spec_name}**：Gaussian Process 擬合完成。")

        rep.append("\n---")

        # 計算重複實驗群組內標準差與良率百分比統計摘要
        rep.append("### 重複實驗群組標準差與製程良率統計")
        total_records = len(df_data)
        unique_groups = df_data['Group_ID'].nunique() if 'Group_ID' in df_data.columns else total_records
        rep.append(f"- **總測試筆數**：{total_records} 筆（對應 {unique_groups} 組獨立控制條件）")

        if 'Group_ID' in df_data.columns:
            rep.append("\n| 實驗群組 | 重複次數 (\(N\)) | 規格項目 | 組內平均值 | 組內標準差 (\(s\)) | 良率狀態 |")
            rep.append("| :--- | :---: | :--- | :---: | :---: | :---: |")
            for grp_id, group_df in df_data.groupby('Group_ID'):
                n_reps = len(group_df)
                for spec_name, spec_type in specs_dict.items():
                    vals = group_df[spec_name].values
                    if spec_type == "continuous":
                        g_mean = np.mean(vals)
                        g_std = np.std(vals, ddof=1) if n_reps > 1 else 0.0
                        cfg = targets_state.get(spec_name, {"target": 100.0, "min_pct": 0.0, "max_pct": 10.0})
                        target = cfg["target"]
                        lower = target * (1.0 - cfg["min_pct"]/100.0)
                        upper = target * (1.0 + cfg["max_pct"]/100.0)
                        is_g_pass = all((vals >= lower) & (vals <= upper))
                        status_str = "合格" if is_g_pass else "異常"
                        rep.append(f"| {grp_id} | {n_reps} | {spec_name} | {g_mean:.2f} | {g_std:.2f} | {status_str} |")
                    else:
                        pass_c = np.sum(vals == 0)
                        rate = (pass_c / n_reps) * 100.0
                        rep.append(f"| {grp_id} | {n_reps} | {spec_name} | - | - | 良率 {rate:.0f}% |")

        rep.append("\n---")
        bounds = [(df_train[col].min(), df_train[col].max()) for col in x_cols]
        init_guess = [(df_train[col].min() + df_train[col].max()) / 2.0 for col in x_cols]

        def robust_loss_function(x_cont):
            x_arr = scaler.transform([x_cont])
            total_loss = 0.0

            for spec_name, info in models.items():
                if info["type"] == "binary":
                    p_fail = info["model"].predict_proba([x_cont])[0][1]
                    if p_fail > binary_caps.get(spec_name, 0.0):
                        total_loss += 10000.0 * (p_fail - binary_caps.get(spec_name, 0.0))
                elif info["type"] == "binary_constant":
                    pass
                else:
                    pred_val, pred_std = info["model"].predict(x_arr, return_std=True)
                    val = pred_val[0]
                    std = pred_std[0]
                    cfg = targets_state.get(spec_name, {"mode": "目標與容忍區間", "target": 100.0, "min_pct": 0.0, "max_pct": 10.0})
                    target = cfg["target"]

                    target_loss = ((val - target) / (target + 1e-5)) ** 2 * 100.0
                    robust_penalty = std * robust_weight
                    total_loss += (target_loss + robust_penalty)

            return total_loss

        res = minimize(robust_loss_function, init_guess, bounds=bounds, method='L-BFGS-B')
        opt_x_cont = res.x
        opt_x_arr = scaler.transform([opt_x_cont])
        closest_idx = np.argmin(np.linalg.norm(df_train[x_cols].values - opt_x_cont, axis=1))
        closest_row = df_train.iloc[closest_idx]
        closest_run_id = closest_row.get('Group_ID', closest_row.get('Run_ID', f'Run #{closest_idx+1}'))

        all_passed = True
        comparison_lines = []
        for spec_name, info in models.items():
            if info["type"] == "continuous":
                p_val, p_std = info["model"].predict(opt_x_arr, return_std=True)
                val = p_val[0]
                cfg = targets_state.get(spec_name, {"mode": "目標與容忍區間", "target": 100.0, "min_pct": 0.0, "max_pct": 10.0})
                target = cfg["target"]
                mode = cfg["mode"]

                if mode == "越高越好":
                    lower_limit, upper_limit = target * (1.0 - cfg["min_pct"]/100.0), float('inf')
                    is_pass = val >= target * (1.0 - cfg["min_pct"]/100.0)
                elif mode == "越低越好":
                    lower_limit, upper_limit = -float('inf'), target * (1.0 + cfg["max_pct"]/100.0)
                    is_pass = val <= target * (1.0 + cfg["max_pct"]/100.0)
                else:
                    lower_limit = target * (1.0 - cfg["min_pct"]/100.0)
                    upper_limit = target * (1.0 + cfg["max_pct"]/100.0)
                    is_pass = (lower_limit <= val <= upper_limit)

                if not is_pass: all_passed = False
                status_str = "符合" if is_pass else "未達標"
                diff = val - target
                comparison_lines.append(f"  - **{spec_name}** [{status_str}] | 預測值: {val:.2f} | 目標值: {target} (範圍: {lower_limit:.2f} ~ {upper_limit:.2f}, 差異: {diff:+.2f}, 標準差: {p_std[0]:.2f})")
            elif info["type"] == "binary_constant":
                p_fail = float(info["constant_val"])
                cap = binary_caps.get(spec_name, 0.0)
                is_pass = p_fail <= cap
                if not is_pass: all_passed = False
                status_str = "符合" if is_pass else "超出"
                comparison_lines.append(f"  - **{spec_name}** [{status_str}] | 常數: {p_fail} (上限: {cap})")
            else:
                p_fail = info["model"].predict_proba([opt_x_cont])[0][1]
                cap = binary_caps.get(spec_name, 0.0)
                is_pass = p_fail <= cap
                if not is_pass: all_passed = False
                status_str = "符合" if is_pass else "超出"
                comparison_lines.append(f"  - **{spec_name}** [{status_str}] | 不良機率: {p_fail*100:.2f}% (上限: {cap*100:.2f}%)")

        max_allowed_std = 1.0
        max_opt_std = max([models[spec_name]["model"].predict(opt_x_arr, return_std=True)[1][0] for spec_name in models if models[spec_name]["type"] == "continuous"], default=0.0)

        is_truly_converged = all_passed and (max_opt_std <= max_allowed_std)

        if is_truly_converged:
            rep.insert(1, f"**收斂狀態**：已達成目標且模型不確定性小於門檻 ({max_opt_std:.3f} <= {max_allowed_std})，停止補點。\n")
            suggestion_df = pd.DataFrame(columns=["Run_ID", "Group_ID", "Replicate_Index"] + x_cols)
            reasoning = "雙重收斂條件已滿足。"
        else:
            rep.insert(1, f"**收斂狀態**：尚未完全滿足收斂條件 (達標: {all_passed}, 最大標準差: {max_opt_std:.3f})，已產生推薦補點。\n")
            suggestion_df, _, reasoning = auto_generate_gpr_robust_sequential_doe(df_train, x_cols, models, batch_size_slider)
            for s_name in y_cols:
                if s_name not in suggestion_df.columns:
                    suggestion_df[s_name] = 0.0 if specs_dict[s_name] == "continuous" else 0

        rep.append(f"### 最佳參數推薦與已知實驗對照")
        rep.append(f"**參考實驗群組**：{closest_run_id}\n")

        rep.append("| 控制變數 | 建議最佳化數值 | 參考實驗值 | 差異 (\(\Delta\)) |")
        rep.append("| :--- | :---: | :---: | :---: |")
        for idx, col in enumerate(x_cols):
            opt_val = opt_x_cont[idx]
            known_val = float(closest_row[col])
            diff_val = opt_val - known_val
            rep.append(f"| **{col}** | {opt_val:.3f} | {known_val:.3f} | {diff_val:+.3f} |")

        rep.append("\n**規格預測與檢驗：**")
        rep.extend(comparison_lines)
        rep.append(f"\n**分析說明**：\n{reasoning}")

        return "\n".join(rep), suggestion_df
    except Exception as e:
        return f"分析過程發生錯誤: {str(e)}", pd.DataFrame()


default_factors = "Spindle_Speed: 1000, 5000 (int); Feed_Rate: 10, 50 (cont); Depth_Cut: 0.1, 0.5 (cont); Pressure: 2, 10 (cont); Temp: 20, 80 (int)"
default_specs = "Depth(µm): continuous : 120; Angle(°): continuous : 45; Damage(0/1): binary : 0"

with gr.Blocks(title="DOE System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 貝氏實驗設計與強健性最佳化系統")

    with gr.Tab("實驗設計與數據"):
        factors_input = gr.Textbox(label="控制因子設定", value=default_factors, lines=3)
        specs_input = gr.Textbox(label="規格與目標設定", value=default_specs, lines=2)
        mode_dropdown = gr.Dropdown(choices=["空間填充設計 (LHS)", "核心因子設計"], value="空間填充設計 (LHS)", label="排程模式")
        replication_checkbox = gr.Checkbox(label="啟用統計學客觀重複次數指派（自動展開多筆測值列）", value=True)

        with gr.Row():
            gen_btn = gr.Button("生成實驗矩陣", variant="primary")
            sim_btn = gr.Button("模擬測試數據", variant="secondary")

        with gr.Row():
            file_input = gr.File(label="上傳 CSV 檔案")
            upload_btn = gr.Button("載入檔案數據", variant="secondary")

        status_output = gr.Markdown()
        matrix_df = gr.Dataframe(interactive=True, label="實驗矩陣工作表（可直接逐筆填寫各重複次數的量測數值）")

    with gr.Tab("模型分析與最佳化"):
        sync_specs_btn = gr.Button("同步介面設定", variant="secondary")
        with gr.Row():
            with gr.Column():
                c1_m, c1_t = gr.Dropdown(choices=["目標與容忍區間", "越高越好", "越低越好"], value="目標與容忍區間", label="連續規格 1 策略"), gr.Number(value=120.0, label="目標值")
                with gr.Row(): c1_min, c1_max = gr.Number(value=0.0, label="下限 (%)"), gr.Number(value=10.0, label="上限 (%)")
            with gr.Column():
                c2_m, c2_t = gr.Dropdown(choices=["目標與容忍區間", "越高越好", "越低越好"], value="目標與容忍區間", label="連續規格 2 策略"), gr.Number(value=45.0, label="目標值")
                with gr.Row(): c2_min, c2_max = gr.Number(value=0.0, label="下限 (%)"), gr.Number(value=10.0, label="上限 (%)")
            with gr.Column():
                c3_m, c3_t = gr.Dropdown(choices=["目標與容忍區間", "越高越好", "越低越好"], value="目標與容忍區間", label="連續規格 3 策略", visible=False), gr.Number(value=100.0, label="目標值", visible=False)
                with gr.Row(): c3_min, c3_max = gr.Number(value=0.0, label="下限 (%)", visible=False), gr.Number(value=10.0, label="上限 (%)", visible=False)

        with gr.Row():
            binary_slider_1 = gr.Slider(0.0, 0.05, 0.0, step=0.001, label="二元分類護欄 1")
            binary_slider_2 = gr.Slider(0.0, 0.05, 0.0, step=0.001, label="二元分類護欄 2", visible=False)
            binary_slider_3 = gr.Slider(0.0, 0.05, 0.0, step=0.001, label="二元分類護欄 3", visible=False)

        with gr.Row():
            robust_weight_slider = gr.Slider(0.0, 50.0, 10.0, step=1.0, label="強健性懲罰權重")
            batch_size_slider = gr.Slider(2, 12, 5, step=1, label="增量補點數量 (Batch Size)")

        with gr.Row():
            analyze_btn = gr.Button("執行分析與最佳化", variant="primary")
            feed_back_btn = gr.Button("將推薦點追加至矩陣", variant="secondary")

        analysis_report_md = gr.Markdown()

        gr.Markdown("### 推薦補點清單")
        suggestion_output_df = gr.Dataframe(label="系統推薦矩陣")

    gen_btn.click(generate_stage1_matrix, inputs=[factors_input, specs_input, mode_dropdown, replication_checkbox], outputs=[matrix_df, c1_m, c1_t, c1_min, c1_max, c2_m, c2_t, c2_min, c2_max, c3_m, c3_t, c3_min, c3_max, binary_slider_1, binary_slider_2, binary_slider_3, status_output])
    sim_btn.click(simulate_data_fill_with_targets, inputs=[matrix_df, specs_input], outputs=[matrix_df, status_output])
    upload_btn.click(upload_real_plant_data, inputs=[file_input, matrix_df], outputs=[matrix_df, status_output])
    sync_specs_btn.click(update_dynamic_specs_ui, inputs=[specs_input], outputs=[c1_m, c1_t, c1_min, c1_max, c2_m, c2_t, c2_min, c2_max, c3_m, c3_t, c3_min, c3_max, binary_slider_1, binary_slider_2, binary_slider_3])

    analyze_btn.click(analyze_multi_objective_doe_robust, inputs=[matrix_df, specs_input, c1_m, c1_t, c1_min, c1_max, c2_m, c2_t, c2_min, c2_max, c3_m, c3_t, c3_min, c3_max, binary_slider_1, binary_slider_2, binary_slider_3, robust_weight_slider, batch_size_slider], outputs=[analysis_report_md, suggestion_output_df])

    feed_back_btn.click(feed_back_suggestions_to_matrix, inputs=[matrix_df, suggestion_output_df, specs_input], outputs=[matrix_df, status_output])

if __name__ == "__main__":
    demo.launch()
