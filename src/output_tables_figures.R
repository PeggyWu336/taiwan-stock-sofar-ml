# ============================================================
# output_tables_figures.R
# 論文圖表輸出腳本
# 自動抓取 pipeline_runs/ 下最新的 run_* 資料夾
# 輸出所有論文所需圖表至 thesis_output/ 資料夾
#
# 對應輸出：
#   表4   財務特徵描述統計（季頻，25項）
#   表5   市場面特徵相關係數矩陣
#   表6   各模型樣本外 R²_OOS
#   表7   各模型投資組合績效
#   表8   跨越檢定結果
#   表9   主要模型逐年夏普比率
#   表10  主要模型逐年 R²_OOS
#   表11  XGBoost-Hybrid 特徵重要性
#   表A.1 各模型毛報酬與淨報酬績效比較
#   表B.1 SOFAR vs PCA 降維方法績效比較
#   圖1   每月 Universe 規模
#   圖2   主要模型累積財富走勢
# ============================================================


# ============================================================
# 0) 套件載入
# ============================================================

pkgs <- c("data.table", "ggplot2", "lubridate")
to_install <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(to_install) > 0) install.packages(to_install)
invisible(lapply(pkgs, library, character.only = TRUE))


# ============================================================
# 1) 自動抓取最新 run 資料夾
# ============================================================

# 所有 run 的根目錄（與 pipeline_main.R 一致）
BASE_OUT_DIR <- "C:/我的/政大/碩二下/論文輸出資料與模型/pipeline_runs"

# 找出所有 run_* 資料夾，取最新（依名稱排序，時間戳記格式保證正確）
run_dirs <- list.dirs(BASE_OUT_DIR, recursive = FALSE, full.names = TRUE)
run_dirs <- run_dirs[grepl("run_\\d{8}_\\d{6}$", basename(run_dirs))]

if (length(run_dirs) == 0) stop("找不到任何 run_* 資料夾，請先執行 pipeline_main.R。")

LATEST_RUN <- tail(sort(run_dirs), 1)  # 字典序最大 = 最新
OUT_DIR_REFIT  <- file.path(LATEST_RUN, "annual_refit")
OUT_DIR_PCA    <- file.path(LATEST_RUN, "pca_comparison")
OUT_DIR_THESIS <- file.path(LATEST_RUN, "thesis_output")

# 確認主要資料夾存在
if (!dir.exists(OUT_DIR_REFIT))
  stop(paste("找不到 annual_refit 資料夾：", OUT_DIR_REFIT))

# 建立論文輸出資料夾（若尚未建立）
dir.create(OUT_DIR_THESIS, recursive = TRUE, showWarnings = FALSE)

cat("========================================\n")
cat("output_tables_figures.R\n")
cat("Latest run :", LATEST_RUN, "\n")
cat("Thesis out :", OUT_DIR_THESIS, "\n")
cat("========================================\n\n")


# ============================================================
# 2) 輔助函數
# ============================================================

# 從資料夾內自動抓取符合 pattern 的最新（唯一）檔案
find_file <- function(dir, pattern) {
  fs <- list.files(dir, pattern = pattern, full.names = TRUE)
  if (length(fs) == 0) return(NULL)
  fs[which.max(file.info(fs)$mtime)]   # 最新修改時間
}

# 讀取多空淨報酬時間序列（從 predsort_FULL_OOS_ 系列檔案）
read_ls_net <- function(model_code) {
  f <- find_file(OUT_DIR_REFIT,
                 paste0("predsort_FULL_OOS_", model_code, "_.*\\.csv"))
  if (is.null(f)) { cat("⚠️ 找不到：", model_code, "\n"); return(NULL) }
  dt <- fread(f)
  dt[, .(ym = as.Date(ym),
         ls_gross = ls_gross,
         ls_net   = ls_net,
         wealth_ls_gross = wealth_ls_gross,
         wealth_ls_net   = wealth_ls_net,
         turnover_ls     = turnover_ls)]
}

# NW t 統計量（lag=6）
nw_tstat <- function(r, lag = 6) {
  r <- r[is.finite(r)]
  if (length(r) < 6) return(NA_real_)
  tryCatch({
    fit <- lm(r ~ 1)
    se  <- sqrt(as.numeric(
      sandwich::NeweyWest(fit, lag = lag, prewhite = FALSE)[1, 1]))
    mean(r) / se
  }, error = function(e) NA_real_)
}

# 年化報酬（幾何平均）
ann_ret <- function(r, freq = 12) {
  r <- r[is.finite(r)]
  if (length(r) < 1) return(NA_real_)
  prod(1 + r)^(freq / length(r)) - 1
}

# Sharpe ratio（年化）
sharpe <- function(r, freq = 12) {
  r <- r[is.finite(r)]
  if (length(r) < 2) return(NA_real_)
  mu  <- mean(r); sdv <- sd(r)
  ann_vol <- sdv * sqrt(freq)
  if (ann_vol == 0) return(NA_real_)
  (mu * freq) / ann_vol
}

# 最大回撤
max_drawdown <- function(r) {
  r <- r[is.finite(r)]
  if (length(r) < 1) return(NA_real_)
  cum <- cumprod(1 + r); peak <- cummax(cum)
  min((cum - peak) / peak, na.rm = TRUE)
}

# 計算單一模型完整績效（毛報酬 + 淨報酬）
calc_perf <- function(model_code) {
  bt <- read_ls_net(model_code)
  if (is.null(bt)) return(NULL)
  r_g <- bt$ls_gross; r_n <- bt$ls_net
  to  <- bt$turnover_ls
  data.table(
    model          = model_code,
    ann_ret_gross  = ann_ret(r_g),
    sharpe_gross   = sharpe(r_g),
    ann_ret_net    = ann_ret(r_n),
    sharpe_net     = sharpe(r_n),
    tstat_nw_net   = nw_tstat(r_n),
    maxdd_net      = max_drawdown(r_n),
    avg_turnover   = mean(to, na.rm = TRUE)
  )
}


# ============================================================
# 3) 讀入基礎資料檔
# ============================================================

# 完整預測 panel（用於計算 R²_OOS）
pred_panel_file <- find_file(OUT_DIR_REFIT, "pred_panel_ANNUALREFIT_ED_.*\\.csv")
if (is.null(pred_panel_file)) stop("找不到 pred_panel_ANNUALREFIT_ED_*.csv")
pred_dt <- fread(pred_panel_file)
cat("預測 panel 載入：", nrow(pred_dt), "列\n")

# 所有預測欄位
pred_cols_all <- grep("^pred_", names(pred_dt), value = TRUE)
# 只取 9 個 ML 模型 + OLS-3（排除 sofar_comp 系列，因為不是報酬預測值）
pred_cols_main <- intersect(
  c("pred_en_S", "pred_rf_S", "pred_xgb_S",
    "pred_en_M", "pred_rf_M", "pred_xgb_M",
    "pred_en_H", "pred_rf_H", "pred_xgb_H",
    "pred_ols3"),
  pred_cols_all
)

# 樣本外年份
years <- sort(unique(pred_dt$year_refit))


# ============================================================
# 4) 表4：財務特徵描述統計
#    來源：pipeline_main.R 獨立計算的 table4_desc_stats_*.csv
# ============================================================

cat("\n--- 表4：財務特徵描述統計 ---\n")

table4_file <- find_file(OUT_DIR_REFIT, "table4_desc_stats_.*\\.csv")
if (!is.null(table4_file)) {
  table4 <- fread(table4_file)

  # 整理欄位格式（四捨五入至論文精度）
  num_cols <- c("mean", "median", "sd", "min", "max")
  table4[, (num_cols) := lapply(.SD, function(x) round(x, 3)), .SDcols = num_cols]
  table4[, na_rate := paste0(round(na_rate * 100, 1), "%")]

  fwrite(table4, file.path(OUT_DIR_THESIS, "Table4_desc_stats.csv"))
  cat("表4 已輸出。\n"); print(table4)
} else {
  cat("⚠️ 找不到 table4_desc_stats_*.csv，跳過。\n")
}


# ============================================================
# 5) 表5：市場面特徵相關係數矩陣
#    來源：pipeline_main.R 獨立計算的 table5_market_corr_*.csv
# ============================================================

cat("\n--- 表5：市場面特徵相關係數矩陣 ---\n")

table5_file <- find_file(OUT_DIR_REFIT, "table5_market_corr_.*\\.csv")
if (!is.null(table5_file)) {
  table5 <- fread(table5_file)
  # 數值欄位四捨五入至小數點後三位
  num_cols5 <- setdiff(names(table5), "variable")
  table5[, (num_cols5) := lapply(.SD, function(x) round(x, 3)),
         .SDcols = num_cols5]
  fwrite(table5, file.path(OUT_DIR_THESIS, "Table5_market_corr.csv"))
  cat("表5 已輸出。\n"); print(table5)
} else {
  cat("⚠️ 找不到 table5_market_corr_*.csv，跳過。\n")
}


# ============================================================
# 6) 表6：各模型樣本外 R²_OOS（%）
#    對應論文公式 (11)，R̄ = 0 基準
# ============================================================

cat("\n--- 表6：全期 R²_OOS ---\n")

r2_full <- sapply(pred_cols_main, function(pc) {
  dt <- pred_dt[is.finite(y) & is.finite(get(pc))]
  if (nrow(dt) == 0) return(NA_real_)
  ss_res <- sum((dt$y - dt[[pc]])^2)
  ss_tot <- sum((dt$y)^2)
  round((1 - ss_res / ss_tot) * 100, 3)
})

# 整理為論文格式（行：特徵集，列：演算法）
table6 <- data.table(
  特徵集     = c("Market-only", "Market-only", "Market-only",
                  "SOFAR-only",  "SOFAR-only",  "SOFAR-only",
                  "Hybrid",      "Hybrid",       "Hybrid",
                  "OLS-3"),
  演算法     = c("Elastic Net", "Random Forest", "XGBoost",
                  "Elastic Net", "Random Forest", "XGBoost",
                  "Elastic Net", "Random Forest", "XGBoost",
                  "OLS"),
  model_code = pred_cols_main,
  R2_OOS_pct = as.numeric(r2_full)
)

fwrite(table6, file.path(OUT_DIR_THESIS, "Table6_R2OOS.csv"))
cat("表6 已輸出。\n"); print(table6)


# ============================================================
# 7) 表7：各模型投資組合績效（含 NW t 統計量與顯著性標記）
#    毛報酬 + 淨報酬
# ============================================================

cat("\n--- 表7：投資組合績效 ---\n")

all_models <- c(pred_cols_main,
                "pred_sofar_comp_main")  # 純 SOFAR 因子排序

perf_list <- lapply(all_models, calc_perf)
table7_raw <- rbindlist(Filter(Negate(is.null), perf_list))

# 加入顯著性標記（基於 NW t 統計量）
table7_raw[, sig := ifelse(abs(tstat_nw_net) >= 2.576, "***",
                    ifelse(abs(tstat_nw_net) >= 1.960, "**",
                    ifelse(abs(tstat_nw_net) >= 1.645, "*", "")))]

# 格式化數值
# 年化報酬轉百分比
table7_raw[, ann_ret_gross := round(ann_ret_gross * 100, 3)]
table7_raw[, ann_ret_net   := round(ann_ret_net   * 100, 3)]
# 其他欄位四捨五入
table7_raw[, sharpe_gross  := round(sharpe_gross,  3)]
table7_raw[, sharpe_net    := round(sharpe_net,    3)]
table7_raw[, tstat_nw_net  := round(tstat_nw_net,  3)]
table7_raw[, maxdd_net     := round(maxdd_net,     3)]
table7_raw[, avg_turnover  := round(avg_turnover,  4)]
table7_raw[, ann_ret_gross := round(ann_ret_gross / 100, 5)]
table7_raw[, ann_ret_net   := round(ann_ret_net   / 100, 5)]

# 重新整理格式（百分比欄位）
table7 <- copy(table7_raw)
table7[, ann_ret_gross_pct := round(ann_ret_gross * 100, 3)]
table7[, ann_ret_net_pct   := round(ann_ret_net   * 100, 3)]
table7[, sharpe_gross      := round(sharpe_gross,  3)]
table7[, sharpe_net        := round(sharpe_net,    3)]
table7[, tstat_nw_net      := round(tstat_nw_net,  3)]
table7[, maxdd_net_pct     := round(maxdd_net * 100, 3)]
table7[, avg_turnover      := round(avg_turnover,  4)]

fwrite(table7[, .(model, ann_ret_gross_pct, sharpe_gross,
                   ann_ret_net_pct, sharpe_net,
                   tstat_nw_net, sig, maxdd_net_pct, avg_turnover)],
       file.path(OUT_DIR_THESIS, "Table7_portfolio_perf.csv"))
cat("表7 已輸出。\n"); print(table7[, .(model, ann_ret_net_pct, sharpe_net,
                                         tstat_nw_net, sig, maxdd_net_pct)])


# ============================================================
# 8) 表8：跨越檢定（Spanning Test）結果
#    來源：spanning_test_result_*.csv
# ============================================================

cat("\n--- 表8：跨越檢定 ---\n")

span_file <- find_file(OUT_DIR_REFIT, "spanning_test_result_.*\\.csv")
if (!is.null(span_file)) {
  table8 <- fread(span_file)
  # 加入顯著性標記
  table8[, sig := ifelse(abs(t_stat) >= 2.576, "***",
                  ifelse(abs(t_stat) >= 1.960, "**",
                  ifelse(abs(t_stat) >= 1.645, "*", "")))]
  table8[, estimate  := round(estimate,  3)]
  table8[, std_error := round(std_error, 3)]
  table8[, t_stat    := round(t_stat,    3)]
  table8[, p_value   := round(p_value,   4)]
  fwrite(table8, file.path(OUT_DIR_THESIS, "Table8_spanning_test.csv"))
  cat("表8 已輸出。\n"); print(table8)
} else {
  cat("⚠️ 找不到 spanning_test_result_*.csv，跳過。\n")
}


# ============================================================
# 9) 表9：主要模型逐年夏普比率（2016–2025）
#    從逐年績效彙總檔讀取
# ============================================================

cat("\n--- 表9：逐年夏普比率 ---\n")

byyear_file <- find_file(OUT_DIR_REFIT, "predsort_summary_BYYEAR_.*\\.csv")
if (!is.null(byyear_file)) {
  byyear_all <- fread(byyear_file)

  # 主要模型：XGB-H、XGB-S、XGB-M、OLS-3、純SOFAR
  main_models_9 <- c("pred_xgb_H", "pred_xgb_S", "pred_xgb_M",
                      "pred_ols3", "pred_sofar_comp_main")
  label_map9 <- c(
    pred_xgb_H          = "XGB-H",
    pred_xgb_S          = "XGB-S",
    pred_xgb_M          = "XGB-M",
    pred_ols3           = "OLS-3",
    pred_sofar_comp_main = "純SOFAR"
  )

  table9_long <- byyear_all[model %in% main_models_9,
                              .(year_refit, model,
                                sharpe = round(sharpe_ls_net, 3))]
  table9_long[, model_label := label_map9[model]]
  table9_wide <- dcast(table9_long, year_refit ~ model_label,
                        value.var = "sharpe")

  # 加入正值年數統計列
  pos_row <- data.table(year_refit = "正值年數",
                         `XGB-H`   = sum(table9_wide$`XGB-H`   > 0, na.rm = TRUE),
                         `XGB-S`   = sum(table9_wide$`XGB-S`   > 0, na.rm = TRUE),
                         `XGB-M`   = sum(table9_wide$`XGB-M`   > 0, na.rm = TRUE),
                         `OLS-3`   = sum(table9_wide$`OLS-3`   > 0, na.rm = TRUE),
                         `純SOFAR` = sum(table9_wide$`純SOFAR` > 0, na.rm = TRUE))
  # 轉為字元型別再合併
  table9_wide[, year_refit := as.character(year_refit)]
  table9_out <- rbind(table9_wide, pos_row, fill = TRUE)

  fwrite(table9_out, file.path(OUT_DIR_THESIS, "Table9_annual_sharpe.csv"))
  cat("表9 已輸出。\n"); print(table9_out)
} else {
  cat("⚠️ 找不到 predsort_summary_BYYEAR_*.csv，跳過。\n")
}


# ============================================================
# 10) 表10：主要模型逐年 R²_OOS（%）
#     來源：r2_oos_byyear_*.csv
# ============================================================

cat("\n--- 表10：逐年 R²_OOS ---\n")

r2_byyear_file <- find_file(OUT_DIR_REFIT, "r2_oos_byyear_.*\\.csv")
if (!is.null(r2_byyear_file)) {
  r2_byyear <- fread(r2_byyear_file)

  main_models_10 <- c("pred_xgb_H", "pred_xgb_S", "pred_xgb_M", "pred_ols3")
  label_map10 <- c(
    pred_xgb_H = "XGB-H",
    pred_xgb_S = "XGB-S",
    pred_xgb_M = "XGB-M",
    pred_ols3  = "OLS-3"
  )

  table10_long <- r2_byyear[model %in% main_models_10,
                              .(year, model, r2_oos)]
  table10_long[, model_label := label_map10[model]]
  table10_wide <- dcast(table10_long, year ~ model_label,
                         value.var = "r2_oos")

  fwrite(table10_wide, file.path(OUT_DIR_THESIS, "Table10_annual_r2oos.csv"))
  cat("表10 已輸出。\n"); print(table10_wide)
} else {
  cat("⚠️ 找不到 r2_oos_byyear_*.csv，跳過。\n")
}


# ============================================================
# 11) 表11：XGBoost-Hybrid 特徵重要性（Gain）
#     來源：xgb_hybrid_importance_*.csv
# ============================================================

cat("\n--- 表11：XGBoost-Hybrid 特徵重要性 ---\n")

imp_file <- find_file(OUT_DIR_REFIT, "xgb_hybrid_importance_.*\\.csv")
if (!is.null(imp_file)) {
  table11 <- fread(imp_file)
  # Gain 轉為百分比
  if ("Gain" %in% names(table11)) {
    table11[, Gain_pct := round(Gain / sum(Gain) * 100, 3)]
    # 加入排名
    setorder(table11, -Gain_pct)
    table11[, rank := .I]
    # 區分特徵類型
    table11[, feature_type := ifelse(grepl("^sofar_f|^z_sofar",
                                           Feature), "SOFAR因子", "市場面")]
  }
  fwrite(table11, file.path(OUT_DIR_THESIS, "Table11_xgb_importance.csv"))
  cat("表11 已輸出。\n"); print(table11)
} else {
  cat("⚠️ 找不到 xgb_hybrid_importance_*.csv，跳過。\n")
}


# ============================================================
# 12) 表A.1：毛報酬與淨報酬完整績效比較
#     來源：predsort_FULL_OOS_summary_*.csv（含毛報酬）
# ============================================================

cat("\n--- 表A.1：毛報酬 vs 淨報酬 ---\n")

full_sum_file <- find_file(OUT_DIR_REFIT, "predsort_FULL_OOS_summary_.*\\.csv")
if (!is.null(full_sum_file)) {
  full_sum <- fread(full_sum_file)

  # 取主要 10 個模型
  full_sum_main <- full_sum[model %in% c(pred_cols_main, "pred_sofar_comp_main")]

  # 格式化
  full_sum_main[, ann_ret_gross_pct := round(ann_ret_gross * 100, 3)]
  full_sum_main[, ann_ret_net_pct   := round(ann_ret_net   * 100, 3)]
  full_sum_main[, sharpe_gross      := round(sharpe_gross,  3)]
  full_sum_main[, sharpe_net        := round(sharpe_net,    3)]
  full_sum_main[, maxdd_net_pct     := round(maxdd_net * 100, 3)]

  tableA1 <- full_sum_main[, .(model, ann_ret_gross_pct, sharpe_gross,
                                 ann_ret_net_pct, sharpe_net, maxdd_net_pct)]
  fwrite(tableA1, file.path(OUT_DIR_THESIS, "TableA1_gross_vs_net.csv"))
  cat("表A.1 已輸出。\n"); print(tableA1)
} else {
  cat("⚠️ 找不到 predsort_FULL_OOS_summary_*.csv，跳過。\n")
}


# ============================================================
# 13) 表B.1：SOFAR vs PCA 降維方法績效比較
#     來源：pca_comparison/ 資料夾
# ============================================================

cat("\n--- 表B.1：SOFAR vs PCA 比較 ---\n")

if (dir.exists(OUT_DIR_PCA)) {
  pca_sum_file <- find_file(OUT_DIR_PCA, "pca_xgb_full_summary_.*\\.csv")
  if (!is.null(pca_sum_file)) {
    pca_sum <- fread(pca_sum_file)

    # 從主流程取 SOFAR 對應模型數字
    if (!is.null(full_sum_file)) {
      sofar_rows <- full_sum[model %in% c("pred_xgb_S", "pred_xgb_H"),
                             .(model, ann_ret_net, sharpe_net,
                               tstat_nw_net = tstat_nw_net,
                               maxdd_net)]
      sofar_rows[, model := ifelse(model == "pred_xgb_S",
                                    "XGBoost-SOFAR", "XGBoost-Hybrid")]
    } else {
      sofar_rows <- data.table()
    }

    pca_sum[, model := ifelse(model == "pred_xgb_PCA_S",
                               "XGBoost-PCA-S", "XGBoost-PCA-H")]

    # 合併
    tableB1 <- rbind(
      sofar_rows[, .(model, ann_ret_net_pct = round(ann_ret_net * 100, 3),
                      sharpe_net = round(sharpe_net, 3),
                      tstat_nw_net = round(tstat_nw_net, 3),
                      maxdd_net_pct = round(maxdd_net * 100, 3))],
      pca_sum[, .(model, ann_ret_net_pct = round(ann_ret_net * 100, 3),
                   sharpe_net = round(sharpe_net, 3),
                   tstat_nw_net = round(tstat_nw_net, 3),
                   maxdd_net_pct = round(maxdd_net * 100, 3))],
      fill = TRUE
    )

    fwrite(tableB1, file.path(OUT_DIR_THESIS, "TableB1_SOFAR_vs_PCA.csv"))
    cat("表B.1 已輸出。\n"); print(tableB1)
  } else {
    cat("⚠️ 找不到 pca_xgb_full_summary_*.csv，跳過表B.1。\n")
  }
} else {
  cat("⚠️ 找不到 pca_comparison/ 資料夾，跳過表B.1。\n")
}


# ============================================================
# 14) 圖1：每月 Universe 規模（對應論文圖1）
#     來源：universe_size_monthly_*.csv
# ============================================================

cat("\n--- 圖1：Universe 規模走勢 ---\n")

univ_file <- find_file(OUT_DIR_REFIT, "universe_size_monthly_.*\\.csv")
if (!is.null(univ_file)) {
  univ_size <- fread(univ_file)
  univ_size[, ym := as.Date(ym)]

  p1 <- ggplot(univ_size, aes(x = ym, y = N)) +
    geom_line(color = "#2C3E50", linewidth = 0.7) +   # 黑線：每月實際值
    geom_smooth(method = "loess", span = 0.15,          # 紅線：趨勢線
                color = "#E74C3C", linewidth = 0.8, se = FALSE) +
    scale_y_continuous(limits = c(600, 1200),
                       breaks = seq(600, 1200, 100)) +
    scale_x_date(date_breaks = "2 years", date_labels = "%Y") +
    labs(x = "年份", y = "股票數（檔）") +
    theme_classic() +
    theme(axis.title = element_text(size = 12),
          axis.text  = element_text(size = 11))

  ggsave(file.path(OUT_DIR_THESIS, "Figure1_universe_size.png"),
         plot = p1, width = 8, height = 5, dpi = 300)
  cat("圖1 已輸出。\n")
} else {
  cat("⚠️ 找不到 universe_size_monthly_*.csv，跳過圖1。\n")
}


# ============================================================
# 15) 圖2：主要模型累積財富走勢（對應論文圖2）
#     來源：predsort_FULL_OOS_pred_*_*.csv
# ============================================================

cat("\n--- 圖2：累積財富走勢 ---\n")

# 讀入各模型的累積財富序列
model_specs <- list(
  list(code  = "pred_xgb_H",
       label = "XGBoost-Hybrid"),
  list(code  = "pred_xgb_S",
       label = "XGBoost-SOFAR"),
  list(code  = "pred_xgb_M",
       label = "XGBoost-Market"),
  list(code  = "pred_ols3",
       label = "OLS-3"),
  list(code  = "pred_sofar_comp_main",
       label = "純SOFAR因子排序")
)

dt_list2 <- lapply(model_specs, function(ms) {
  f <- find_file(OUT_DIR_REFIT,
                 paste0("predsort_FULL_OOS_", ms$code, "_.*\\.csv"))
  if (is.null(f)) { cat("⚠️ 找不到：", ms$code, "\n"); return(NULL) }
  dt <- fread(f)
  dt[, .(ym = as.Date(ym), cum_ret = wealth_ls_net, model = ms$label)]
})
cum_dt <- rbindlist(Filter(Negate(is.null), dt_list2))

# 強制 factor 順序（控制圖例排列）
model_levels <- c("XGBoost-Hybrid", "XGBoost-SOFAR", "XGBoost-Market",
                   "OLS-3", "純SOFAR因子排序")
cum_dt[, model := factor(model, levels = model_levels)]

# 顏色、線條樣式、線條粗細設定
colors <- c(
  "XGBoost-Hybrid"  = "#2C3E50",
  "XGBoost-SOFAR"   = "#E74C3C",
  "XGBoost-Market"  = "#3498DB",
  "OLS-3"           = "#27AE60",
  "純SOFAR因子排序" = "#F39C12"
)
linetypes <- c(
  "XGBoost-Hybrid"  = "solid",
  "XGBoost-SOFAR"   = "dashed",
  "XGBoost-Market"  = "dotdash",
  "OLS-3"           = "dotted",
  "純SOFAR因子排序" = "longdash"
)
sizes <- c(
  "XGBoost-Hybrid"  = 1.2,   # 最佳模型較粗
  "XGBoost-SOFAR"   = 0.7,
  "XGBoost-Market"  = 0.7,
  "OLS-3"           = 0.7,
  "純SOFAR因子排序" = 0.7
)

p2 <- ggplot(cum_dt, aes(x = ym, y = cum_ret,
                          color    = model,
                          linetype = model,
                          linewidth = model)) +
  geom_line() +
  # 三條事件標記垂直線
  geom_vline(xintercept = as.Date("2020-03-01"),
             linetype = "dashed", color = "grey60", linewidth = 0.5) +
  geom_vline(xintercept = as.Date("2021-05-01"),
             linetype = "dashed", color = "grey60", linewidth = 0.5) +
  geom_vline(xintercept = as.Date("2025-04-01"),
             linetype = "dashed", color = "grey60", linewidth = 0.5) +
  # 事件標記文字
  annotate("text", x = as.Date("2020-03-01"), y = 0.5,
           label = "2020/3\n新冠疫情\n全球股災",
           size = 4.2, color = "grey40", lineheight = 0.9, hjust = 0.5) +
  annotate("text", x = as.Date("2021-05-01"), y = 0.5,
           label = "2021/5\n台灣本土\n疫情爆發",
           size = 4.2, color = "grey40", lineheight = 0.9, hjust = 0.5) +
  annotate("text", x = as.Date("2025-04-01"), y = 0.5,
           label = "2025/4\n川普關稅\n衝擊",
           size = 4.2, color = "grey40", lineheight = 0.9, hjust = 1.1) +
  scale_color_manual(values = colors,    breaks = model_levels) +
  scale_linetype_manual(values = linetypes, breaks = model_levels) +
  scale_linewidth_manual(values = sizes,  breaks = model_levels) +
  scale_y_continuous(limits = c(0.4, NA)) +
  scale_x_date(date_breaks = "2 years", date_labels = "%Y",
               limits = c(as.Date("2016-01-01"), as.Date("2026-03-01"))) +
  labs(x = "年份", y = "累積財富（初始=1）",
       color = NULL, linetype = NULL, linewidth = NULL) +
  theme_classic() +
  theme(
    legend.position = "bottom",
    legend.text     = element_text(size = 10),
    axis.title      = element_text(size = 12),
    axis.text       = element_text(size = 11),
    plot.margin     = margin(10, 10, 10, 10)
  ) +
  guides(color     = guide_legend(nrow = 2),
         linetype  = guide_legend(nrow = 2),
         linewidth = guide_legend(nrow = 2))

ggsave(file.path(OUT_DIR_THESIS, "Figure2_cumret_wealth.png"),
       plot = p2, width = 10, height = 6, dpi = 300)
cat("圖2 已輸出。\n")


# ============================================================
# 16) 完成報告
# ============================================================

# 列出所有已輸出的論文圖表檔案
thesis_files <- list.files(OUT_DIR_THESIS, full.names = FALSE)
cat("\n========================================\n")
cat("✅ output_tables_figures.R 全部完成\n")
cat("論文輸出資料夾：", OUT_DIR_THESIS, "\n")
cat("已輸出檔案：\n")
for (f in thesis_files) cat("  -", f, "\n")
cat("========================================\n")
