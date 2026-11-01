import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

st.set_page_config(
    page_title="二値分類スコア計算ツール (期間別・全期間対応)",
    page_icon="📊",
    layout="wide",
)

st.title("📊 機械学習 二値分類スコア計算ツール")
st.caption("マナビDXQuest2026 Theme3 演習03 / 全期間まとめ & 期間別（1000人×6期間等）の両方に対応")

# --- ファイルアップロード ---
col_up1, col_up2 = st.columns(2)

with col_up1:
    st.subheader("1. 正解データ CSV")
    st.caption("形式: 1行目タイトル行 / 1列目: ID, 2列目: y (0 または 1)")
    gt_file = st.file_uploader(
        "正解データCSVをアップロード",
        type=["csv"],
        key="gt_file"
    )

with col_up2:
    st.subheader("2. 投稿データ CSV")
    st.caption("形式: タイトル行なし / 1列目: ID, 2列目: 予測確率 (0.0 〜 1.0)")
    sub_file = st.file_uploader(
        "投稿データCSVをアップロード",
        type=["csv"],
        key="sub_file"
    )

# --- 両ファイルがアップロードされた場合の処理 ---
if gt_file is not None and sub_file is not None:
    try:
        # 正解データ読み込み（タイトル行あり）
        df_gt = pd.read_csv(gt_file, header=0)
        # 投稿データ読み込み（タイトル行なし）
        df_sub = pd.read_csv(sub_file, header=None)
        
        # 列数チェック
        if df_gt.shape[1] < 2:
            st.error("❌ 正解データCSVには少なくとも2列（ID列, y列）が必要です。")
            st.stop()
        if df_sub.shape[1] < 2:
            st.error("❌ 投稿データCSVには少なくとも2列（ID列, 予測確率列）が必要です。")
            st.stop()

        # 固定形式で抽出（1列目: ID, 2列目: y / 予測値）
        gt_clean = pd.DataFrame({
            "id": df_gt.iloc[:, 0].astype(str).str.strip(),
            "y_true": pd.to_numeric(df_gt.iloc[:, 1], errors="coerce")
        })
        
        sub_clean = pd.DataFrame({
            "id": df_sub.iloc[:, 0].astype(str).str.strip(),
            "y_pred_prob": pd.to_numeric(df_sub.iloc[:, 1], errors="coerce")
        })

        # 行数の一致確認
        if len(gt_clean) != len(sub_clean):
            st.warning(
                f"⚠️ 行数が一致していません（正解データ: {len(gt_clean):,} 行, 投稿データ: {len(sub_clean):,} 行）。"
            )

        # 結合処理（同一順で連続して並んでいる構造を考慮）
        # 1. 行数が同じで、IDの並び順が一致していれば行順で紐付け
        # 2. それ以外の場合は出現順(cumcount)を考慮したID結合
        if len(gt_clean) == len(sub_clean) and (gt_clean["id"].values == sub_clean["id"].values).all():
            merged = pd.DataFrame({
                "id": gt_clean["id"].values,
                "y_true": gt_clean["y_true"].values,
                "y_pred_prob": sub_clean["y_pred_prob"].values,
            })
        else:
            # IDが各期間で繰り返し現れる場合に対応するため、出現番号を付与してマージ
            gt_clean["_idx"] = gt_clean.groupby("id").cumcount()
            sub_clean["_idx"] = sub_clean.groupby("id").cumcount()
            merged = pd.merge(gt_clean, sub_clean, on=["id", "_idx"], how="inner").drop(columns=["_idx"])

            if len(merged) == 0:
                st.error("❌ 正解データと投稿データで一致するIDが1件もありませんでした。IDの形式を確認してください。")
                st.stop()
            elif len(merged) < len(gt_clean) or len(merged) < len(sub_clean):
                st.warning(
                    f"⚠️ IDの一致件数: {len(merged):,} 件（正解データ: {len(gt_clean):,} 件, 投稿データ: {len(sub_clean):,} 件）"
                    " ※共通する行のみを対象に評価します。"
                )

        # 欠損値チェック
        if merged["y_true"].isna().any():
            st.error("❌ 正解データ(y)に欠損または数値以外の値が含まれています。")
            st.stop()
        if merged["y_pred_prob"].isna().any():
            st.error("❌ 投稿データの予測確率に欠損または数値以外の値が含まれています。タイトル行が含まれていないか確認してください。")
            st.stop()

        # 正解ラベルの二値性チェック
        unique_labels = sorted(merged["y_true"].unique())
        if not set(unique_labels).issubset({0, 1}):
            st.error(f"❌ 正解データ(y)は 0 または 1 である必要があります。（検出された値: {unique_labels}）")
            st.stop()
        if len(unique_labels) < 2:
            st.error("❌ 正解データにクラスが1種類しか含まれていません。二値分類のスコアを計算するには 0 と 1 の両方が必要です。")
            st.stop()

        total_rows = len(merged)
        
        # 1期間あたり1000件で固定分割
        period_size = 1000
        num_periods = max(1, int(np.ceil(total_rows / period_size)))

        # 期間ラベルの付与
        period_labels = []
        for i in range(total_rows):
            p_num = (i // period_size) + 1
            period_labels.append(f"期間 {p_num}")
        merged["period"] = period_labels

        # 予測確率のクリッピング（Log Loss計算時のlog(0)対策）
        merged["y_pred_prob_clipped"] = np.clip(merged["y_pred_prob"].values, 1e-15, 1.0 - 1e-15)

        # 評価関数ヘルパー
        def calc_metrics(df_sub_data):
            y_t = df_sub_data["y_true"].values.astype(int)
            y_p = df_sub_data["y_pred_prob"].values.astype(float)
            y_p_clip = df_sub_data["y_pred_prob_clipped"].values.astype(float)
            
            n_samples = len(y_t)
            pos_count = int(y_t.sum())
            pos_ratio = float(y_t.mean())
            
            if len(np.unique(y_t)) < 2:
                # クラスが1種類しかない場合はROC/PR-AUCが計算不能
                roc = np.nan
                pr = np.nan
            else:
                roc = roc_auc_score(y_t, y_p)
                pr = average_precision_score(y_t, y_p)
                
            loss = log_loss(y_t, y_p_clip)
            return {
                "件数": n_samples,
                "陽性数 (比率)": f"{pos_count} ({pos_ratio*100:.1f}%)",
                "ROC-AUC": roc,
                "PR-AUC": pr,
                "Log Loss": loss,
            }

        # --- 全期間 & 期間別スコアの集計 ---
        summary_records = []
        
        # 1. 全期間
        overall_m = calc_metrics(merged)
        summary_records.append({
            "区分": "全期間 (全体)",
            **overall_m
        })
        
        # 2. 各期間
        for p_idx in range(1, num_periods + 1):
            p_name = f"期間 {p_idx}"
            sub_df_p = merged[merged["period"] == p_name]
            p_m = calc_metrics(sub_df_p)
            summary_records.append({
                "区分": p_name,
                **p_m
            })

        # 期間1からの累積期間も比較対象に追加
        for end_period in range(2, min(5, num_periods) + 1):
            period_names = [f"期間 {p_idx}" for p_idx in range(1, end_period + 1)]
            combined_label = "+".join(period_names)
            combined_df = merged[merged["period"].isin(period_names)]
            combined_m = calc_metrics(combined_df)
            summary_records.append({
                "区分": combined_label,
                **combined_m
            })
            
        summary_df = pd.DataFrame(summary_records)

        # --- スコアサマリー表示 ---
        st.divider()
        st.header("🏆 評価指標スコア サマリー")

        # 全期間のメトリクスカード
        st.markdown("### 🌐 全期間 (全体) スコア")
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric(
                label="ROC-AUC (全期間)",
                value=f"{overall_m['ROC-AUC']:.5f}" if not np.isnan(overall_m['ROC-AUC']) else "N/A",
                help="受信者動作特性曲線下面積（1.0に近いほど高精度）"
            )
        with m_col2:
            st.metric(
                label="PR-AUC (全期間)",
                value=f"{overall_m['PR-AUC']:.5f}" if not np.isnan(overall_m['PR-AUC']) else "N/A",
                help="適合率-再現率曲線下面積（不均衡データで重要、1.0に近いほど高精度）"
            )
        with m_col3:
            st.metric(
                label="Log Loss (全期間)",
                value=f"{overall_m['Log Loss']:.5f}",
                help="対数損失（0.0に近いほど高精度）"
            )

        # 全期間 & 各期間の一覧テーブル
        st.markdown("### 📋 全期間 & 各期間のスコア一覧比較")
        # 表示用にフォーマット
        display_summary = summary_df.copy()
        display_summary["ROC-AUC"] = display_summary["ROC-AUC"].apply(lambda v: f"{v:.5f}" if pd.notna(v) else "N/A")
        display_summary["PR-AUC"] = display_summary["PR-AUC"].apply(lambda v: f"{v:.5f}" if pd.notna(v) else "N/A")
        display_summary["Log Loss"] = display_summary["Log Loss"].apply(lambda v: f"{v:.5f}" if pd.notna(v) else "N/A")
        st.dataframe(display_summary, width="stretch", hide_index=True)

        # 期間推移グラフ
        if num_periods > 1:
            st.markdown("### 📈 期間ごとのスコア推移")
            single_period_labels = [f"期間 {i}" for i in range(1, num_periods + 1)]
            period_only_df = summary_df[summary_df["区分"].isin(single_period_labels)].copy()
            
            p_chart_col1, p_chart_col2 = st.columns(2)
            with p_chart_col1:
                fig_auc = go.Figure()
                fig_auc.add_trace(go.Scatter(
                    x=period_only_df["区分"],
                    y=period_only_df["ROC-AUC"],
                    mode="lines+markers",
                    name="ROC-AUC",
                    line=dict(color="#1f77b4", width=2.5),
                    marker=dict(size=8)
                ))
                fig_auc.add_trace(go.Scatter(
                    x=period_only_df["区分"],
                    y=period_only_df["PR-AUC"],
                    mode="lines+markers",
                    name="PR-AUC",
                    line=dict(color="#ff7f0e", width=2.5),
                    marker=dict(size=8)
                ))
                # 全期間のスコアを水平破線で表示
                if not np.isnan(overall_m['ROC-AUC']):
                    fig_auc.add_hline(y=overall_m['ROC-AUC'], line_dash="dash", line_color="#1f77b4", opacity=0.5, annotation_text="全期間 ROC-AUC")
                if not np.isnan(overall_m['PR-AUC']):
                    fig_auc.add_hline(y=overall_m['PR-AUC'], line_dash="dash", line_color="#ff7f0e", opacity=0.5, annotation_text="全期間 PR-AUC")
                    
                fig_auc.update_layout(
                    title="期間別 ROC-AUC / PR-AUC 推移",
                    xaxis_title="期間",
                    yaxis_title="スコア",
                    height=380,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_auc, width="stretch")

            with p_chart_col2:
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    x=period_only_df["区分"],
                    y=period_only_df["Log Loss"],
                    mode="lines+markers",
                    name="Log Loss",
                    line=dict(color="#d62728", width=2.5),
                    marker=dict(size=8)
                ))
                fig_loss.add_hline(y=overall_m['Log Loss'], line_dash="dash", line_color="#d62728", opacity=0.5, annotation_text="全期間 Log Loss")
                fig_loss.update_layout(
                    title="期間別 Log Loss 推移 (低いほど良い)",
                    xaxis_title="期間",
                    yaxis_title="Log Loss",
                    height=380,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_loss, width="stretch")

        # --- 対象期間の選択 & 詳細分析 ---
        st.divider()
        st.header("🔍 詳細分析（期間切り替え）")
        
        single_period_options = [f"期間 {i}" for i in range(1, num_periods + 1)]
        period_selection_map = {period: [period] for period in single_period_options}
        for end_period in range(2, min(5, num_periods) + 1):
            combined_label = "+".join(f"期間 {i}" for i in range(1, end_period + 1))
            period_selection_map[combined_label] = [
                f"期間 {i}" for i in range(1, end_period + 1)
            ]
        target_options = ["全期間 (全体)"] + list(period_selection_map)
        selected_period = st.selectbox(
            "分析対象の期間を選択してください",
            target_options,
            index=0,
            help="選択した期間のROC/PR曲線、混同行列、予測値分布を確認できます。"
        )

        # 選択されたデータの抽出
        if selected_period == "全期間 (全体)":
            detail_data = merged.copy()
        else:
            detail_data = merged[merged["period"].isin(period_selection_map[selected_period])].copy()

        y_true_detail = detail_data["y_true"].values.astype(int)
        y_pred_detail = detail_data["y_pred_prob"].values.astype(float)
        
        # --- 各指標の最適閾値（一番良い値になる閾値）の探索（0.01刻み 0.01〜0.99） ---
        grid_ths = np.round(np.arange(0.01, 1.00, 0.01), 2)
        y_pred_mat = y_pred_detail[None, :] >= grid_ths[:, None]
        y_true_bool = y_true_detail.astype(bool)[None, :]
        
        tp_grid = np.sum(y_pred_mat & y_true_bool, axis=1)
        fp_grid = np.sum(y_pred_mat & ~y_true_bool, axis=1)
        fn_grid = np.sum(~y_pred_mat & y_true_bool, axis=1)
        tn_grid = np.sum(~y_pred_mat & ~y_true_bool, axis=1)
        
        accs_grid = (tp_grid + tn_grid) / len(y_true_detail)
        
        denom_tpr = tp_grid + fn_grid
        tprs_grid = np.divide(tp_grid, denom_tpr, out=np.zeros_like(tp_grid, dtype=float), where=(denom_tpr > 0))
        
        denom_fpr = fp_grid + tn_grid
        fprs_grid = np.divide(fp_grid, denom_fpr, out=np.zeros_like(fp_grid, dtype=float), where=(denom_fpr > 0))
        
        denom_prec = tp_grid + fp_grid
        precs_grid = np.divide(tp_grid, denom_prec, out=np.zeros_like(tp_grid, dtype=float), where=(denom_prec > 0))
        
        recs_grid = tprs_grid
        denom_f1 = precs_grid + recs_grid
        f1s_grid = np.divide(2 * precs_grid * recs_grid, denom_f1, out=np.zeros_like(denom_f1, dtype=float), where=(denom_f1 > 0))
        
        # 1. Accuracy最良 (最大値)
        cand_acc = np.where(accs_grid == np.max(accs_grid))[0]
        opt_th_acc = float(grid_ths[cand_acc[np.lexsort((-f1s_grid[cand_acc], np.abs(grid_ths[cand_acc] - 0.5)))[0]]])
        
        # 2. TPR最良 (最大値の中でFPR最小・閾値最高)
        cand_tpr = np.where(tprs_grid == np.max(tprs_grid))[0]
        opt_th_tpr = float(grid_ths[cand_tpr[np.lexsort((-grid_ths[cand_tpr], fprs_grid[cand_tpr]))[0]]])
        
        # 3. FPR最良 (最小値の中でTPR最大・閾値最低)
        cand_fpr = np.where(fprs_grid == np.min(fprs_grid))[0]
        opt_th_fpr = float(grid_ths[cand_fpr[np.lexsort((grid_ths[cand_fpr], -tprs_grid[cand_fpr]))[0]]])
        
        # 4. Precision最良 (最大値の中でRecall最大)
        cand_prec = np.where(precs_grid == np.max(precs_grid))[0]
        opt_th_prec = float(grid_ths[cand_prec[np.lexsort((grid_ths[cand_prec], -recs_grid[cand_prec]))[0]]])
        
        # 5. Recall最良 (最大値の中でPrecision最大)
        cand_rec = np.where(recs_grid == np.max(recs_grid))[0]
        opt_th_rec = float(grid_ths[cand_rec[np.lexsort((-grid_ths[cand_rec], -precs_grid[cand_rec]))[0]]])
        
        # 6. F1-Score最良 (最大値)
        cand_f1 = np.where(f1s_grid == np.max(f1s_grid))[0]
        opt_th_f1 = float(grid_ths[cand_f1[np.argmin(np.abs(grid_ths[cand_f1] - 0.5))]])

        # スライダー用セッションステートの初期化 & コールバック
        slider_key = f"th_slider_{selected_period}"
        if slider_key not in st.session_state:
            st.session_state[slider_key] = 0.50

        def set_th_val(target_key, val):
            st.session_state[target_key] = float(np.round(val, 2))

        # 閾値シミュレーション
        st.subheader(f"🎯 閾値シミュレーション（{selected_period}）")
        threshold = st.slider(
            f"二値判定閾値 ({selected_period})",
            min_value=0.01,
            max_value=0.99,
            step=0.01,
            key=slider_key,
        )
        y_pred_binary = (y_pred_detail >= threshold).astype(int)

        acc = accuracy_score(y_true_detail, y_pred_binary)
        prec = precision_score(y_true_detail, y_pred_binary, zero_division=0)
        rec = recall_score(y_true_detail, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_detail, y_pred_binary, zero_division=0)

        # 混同行列からTPR (真陽性率) / FPR (偽陽性率) を算出
        cm_sim = confusion_matrix(y_true_detail, y_pred_binary, labels=[0, 1])
        tn_sim, fp_sim, fn_sim, tp_sim = cm_sim.ravel()
        tpr_val = tp_sim / (tp_sim + fn_sim) if (tp_sim + fn_sim) > 0 else 0.0
        fpr_val = fp_sim / (fp_sim + tn_sim) if (fp_sim + tn_sim) > 0 else 0.0

        s_col1, s_col2, s_col3, s_col4, s_col5, s_col6 = st.columns(6)
        with s_col1:
            st.metric(
                "Accuracy (正解率)",
                f"{acc:.4f}",
                help="(TP + TN) / (TP + TN + FP + FN) ：全データのうち正しく予測できた割合"
            )
            st.button(
                f"🎯 最適化 ({opt_th_acc:.2f})",
                key=f"btn_acc_{selected_period}",
                on_click=set_th_val,
                args=(slider_key, opt_th_acc),
                width="stretch",
                help=f"Accuracyが最大（{np.max(accs_grid):.4f}）となる閾値 {opt_th_acc:.2f} に設定"
            )

        with s_col2:
            st.metric(
                "TPR (真陽性率)",
                f"{tpr_val:.4f}",
                help="TP / (TP + FN) ：真陽性率（実際の陽性のうち正しく検出できた割合、再現率と同義）"
            )
            st.button(
                f"🎯 最適化 ({opt_th_tpr:.2f})",
                key=f"btn_tpr_{selected_period}",
                on_click=set_th_val,
                args=(slider_key, opt_th_tpr),
                width="stretch",
                help=f"TPRが最大（{np.max(tprs_grid):.4f}）となる閾値 {opt_th_tpr:.2f} に設定"
            )

        with s_col3:
            st.metric(
                "FPR (偽陽性率)",
                f"{fpr_val:.4f}",
                help="FP / (FP + TN) ：偽陽性率（実際の陰性のうち誤って陽性と判定してしまった割合）"
            )
            st.button(
                f"🎯 最適化 ({opt_th_fpr:.2f})",
                key=f"btn_fpr_{selected_period}",
                on_click=set_th_val,
                args=(slider_key, opt_th_fpr),
                width="stretch",
                help=f"FPRが最小（{np.min(fprs_grid):.4f}）となる閾値 {opt_th_fpr:.2f} に設定"
            )

        with s_col4:
            st.metric(
                "Precision (適合率)",
                f"{prec:.4f}",
                help="TP / (TP + FP) ：陽性と予測した中で実際に陽性だった割合"
            )
            st.button(
                f"🎯 最適化 ({opt_th_prec:.2f})",
                key=f"btn_prec_{selected_period}",
                on_click=set_th_val,
                args=(slider_key, opt_th_prec),
                width="stretch",
                help=f"Precisionが最大（{np.max(precs_grid):.4f}）となる閾値 {opt_th_prec:.2f} に設定"
            )

        with s_col5:
            st.metric(
                "Recall (再現率)",
                f"{rec:.4f}",
                help="TP / (TP + FN) ：実際の陽性のうち正しく陽性と判定できた割合（= TPR）"
            )
            st.button(
                f"🎯 最適化 ({opt_th_rec:.2f})",
                key=f"btn_rec_{selected_period}",
                on_click=set_th_val,
                args=(slider_key, opt_th_rec),
                width="stretch",
                help=f"Recallが最大（{np.max(recs_grid):.4f}）となる閾値 {opt_th_rec:.2f} に設定"
            )

        with s_col6:
            st.metric(
                "F1-Score",
                f"{f1:.4f}",
                help="2 * (Precision * Recall) / (Precision + Recall) ：適合率と再現率の調和平均"
            )
            st.button(
                f"🎯 最適化 ({opt_th_f1:.2f})",
                key=f"btn_f1_{selected_period}",
                on_click=set_th_val,
                args=(slider_key, opt_th_f1),
                width="stretch",
                help=f"F1-Scoreが最大（{np.max(f1s_grid):.4f}）となる閾値 {opt_th_f1:.2f} に設定"
            )

        # 詳細タブ表示
        tab1, tab2, tab3 = st.tabs(["ROC曲線 & PR曲線", "混同行列 (Confusion Matrix)", "予測確率分布 & データ一覧"])

        with tab1:
            if len(np.unique(y_true_detail)) < 2:
                st.warning("⚠️ この期間の正解データにはクラスが1種類しかないため、ROC曲線/PR曲線を描画できません。")
            else:
                g_col1, g_col2 = st.columns(2)
                curr_roc = roc_auc_score(y_true_detail, y_pred_detail)
                curr_pr = average_precision_score(y_true_detail, y_pred_detail)

                # ROC曲線
                fpr, tpr, _ = roc_curve(y_true_detail, y_pred_detail)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {curr_roc:.4f})', line=dict(color='#1f77b4', width=2)))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='ランダム (AUC = 0.5)', line=dict(color='gray', dash='dash')))
                fig_roc.add_trace(go.Scatter(
                    x=[fpr_val], y=[tpr_val], mode='markers',
                    name=f'現在閾値 ({threshold:.2f}): FPR={fpr_val:.3f}, TPR={tpr_val:.3f}',
                    marker=dict(color='red', size=10, symbol='circle')
                ))
                fig_roc.update_layout(
                    title=f'ROC 曲線 ({selected_period})',
                    xaxis_title='偽陽性率 (False Positive Rate / FPR)',
                    yaxis_title='真陽性率 (True Positive Rate / TPR)',
                    xaxis=dict(range=[-0.02, 1.02]),
                    yaxis=dict(range=[-0.02, 1.02]),
                    height=420,
                )
                g_col1.plotly_chart(fig_roc, width="stretch")

                # PR曲線
                precision_vals, recall_vals, _ = precision_recall_curve(y_true_detail, y_pred_detail)
                baseline = float(y_true_detail.mean())
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(x=recall_vals, y=precision_vals, mode='lines', name=f'PR (AUC = {curr_pr:.4f})', line=dict(color='#ff7f0e', width=2)))
                fig_pr.add_trace(go.Scatter(x=[0, 1], y=[baseline, baseline], mode='lines', name=f'ベースライン ({baseline:.2f})', line=dict(color='gray', dash='dash')))
                fig_pr.add_trace(go.Scatter(
                    x=[rec], y=[prec], mode='markers',
                    name=f'現在閾値 ({threshold:.2f}): Recall={rec:.3f}, Precision={prec:.3f}',
                    marker=dict(color='red', size=10, symbol='circle')
                ))
                fig_pr.update_layout(
                    title=f'Precision-Recall (PR) 曲線 ({selected_period})',
                    xaxis_title='再現率 (Recall / TPR)',
                    yaxis_title='適合率 (Precision)',
                    xaxis=dict(range=[-0.02, 1.02]),
                    yaxis=dict(range=[-0.02, 1.02]),
                    height=420,
                )
                g_col2.plotly_chart(fig_pr, width="stretch")

        with tab2:
            cm = confusion_matrix(y_true_detail, y_pred_binary)
            cm_labels = ["正解: 0", "正解: 1"]
            pred_labels = ["予測: 0", "予測: 1"]

            cm_col1, cm_col2 = st.columns([1, 1])
            with cm_col1:
                st.write(f"**混同行列（{selected_period} / 閾値 = {threshold:.2f}）**")
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    labels=dict(x="予測ラベル", y="正解ラベル", color="件数"),
                    x=pred_labels,
                    y=cm_labels,
                    color_continuous_scale="Blues"
                )
                fig_cm.update_layout(height=380)
                st.plotly_chart(fig_cm, width="stretch")

            with cm_col2:
                tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
                st.markdown(f"### 内訳詳細 ({selected_period})")
                st.write(f"- **True Negative (TN, 陰性を正しく予測)**: {tn:,} 件")
                st.write(f"- **False Positive (FP, 偽陽性・誤検知)**: {fp:,} 件")
                st.write(f"- **False Negative (FN, 偽陰性・見逃し)**: {fn:,} 件")
                st.write(f"- **True Positive (TP, 陽性を正しく予測)**: {tp:,} 件")
                st.markdown(f"**陽性比率（正解）**: {y_true_detail.mean()*100:.1f}%")
                st.markdown(f"**陽性比率（予測）**: {y_pred_binary.mean()*100:.1f}%")

        with tab3:
            st.markdown(f"### 正解ラベル別の予測確率分布 ({selected_period})")
            fig_hist = px.histogram(
                detail_data,
                x="y_pred_prob",
                color=detail_data["y_true"].astype(str),
                barmode="overlay",
                nbins=30,
                opacity=0.6,
                labels={"y_pred_prob": "予測確率", "color": "正解ラベル(y)"},
                title=f"正解ラベルごとの予測確率ヒストグラム ({selected_period})",
                color_discrete_map={"0": "#1f77b4", "1": "#e377c2"}
            )
            fig_hist.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text=f"閾値 {threshold:.2f}")
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, width="stretch")

            st.markdown(f"### データ一覧（誤差確認: {selected_period}）")
            detail_data["y_pred_binary"] = y_pred_binary
            detail_data["is_correct"] = (detail_data["y_true"] == detail_data["y_pred_binary"])
            detail_data["abs_error"] = (detail_data["y_true"] - detail_data["y_pred_prob"]).abs()

            show_only_errors = st.checkbox("誤判定したデータのみ表示", key=f"err_{selected_period}")
            view_df = detail_data[~detail_data["is_correct"]] if show_only_errors else detail_data
            st.dataframe(
                view_df[["id", "period", "y_true", "y_pred_prob", "y_pred_binary", "is_correct", "abs_error"]]
                .sort_values(by="abs_error", ascending=False),
                width="stretch"
            )

        # 評価結果のCSVダウンロード（全期間分・期間列付き）
        st.divider()
        merged["y_pred_binary"] = (merged["y_pred_prob"] >= threshold).astype(int)
        merged["is_correct"] = (merged["y_true"] == merged["y_pred_binary"])
        merged["abs_error"] = (merged["y_true"] - merged["y_pred_prob"]).abs()
        
        csv_data = merged[["id", "period", "y_true", "y_pred_prob", "y_pred_binary", "is_correct", "abs_error"]].to_csv(index=False).encode('utf-8_sig')
        st.download_button(
            "📥 全期間の結合評価データ（期間列付き）をCSVダウンロード",
            data=csv_data,
            file_name="evaluation_result_with_periods.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ 処理中にエラーが発生しました: {e}")

else:
    st.info("💡 左側（正解データCSV: 6000件）と右側（投稿データCSV: 6000件）の両方をアップロードすると、全期間および期間別のスコアが即座に算出されます。")
