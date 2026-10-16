# ============================================================
# pipeline_main.R
# 論文主流程：SOFAR + 機器學習 + PCA 對照實驗
# 對應論文章節：第三章（研究方法）、第四章（實證資料）、第五章（實證結果）
#
# 執行順序：
#   0)  套件載入
#   1)  路徑設定與資料夾建立
#   2)  讀資料 + QC
#   3)  市場面特徵計算
#   4)  Universe flag
#   5)  表4 描述統計（原始資料，補值前，獨立計算）
#   6)  表5 市場面特徵相關係數矩陣（獨立計算）
#   7)  圖1 Universe 規模走勢
#   8)  SOFAR 候選變數白名單
#   9)  Helper functions
#   10) NA tracking：訓練期擴展窗格（2005–2015）
#   11) Annual Refitting Loop（SOFAR → ML → 回測）
#   12) FULL OOS 彙總 + 逐年 R²_OOS
#   13) 符號修正穩健性（8D）
#   14) XGBoost-Hybrid 特徵重要性
#   15) PCA 對照實驗（附錄 B）
#   16) 全部結果輸出至 run_* 資料夾
# ============================================================


# ============================================================
# 0) 套件載入
# ============================================================

pkgs <- c(
  "data.table",   # 高效資料處理
  "lubridate",    # 日期處理
  "zoo",          # 時間序列輔助
  "rrpack",       # SOFAR 估計
  "glmnet",       # Elastic Net
  "ranger",       # Random Forest
  "xgboost",      # XGBoost
  "sandwich",     # Newey-West 標準誤
  "lmtest",       # coeftest（跨越檢定用）
  "ggplot2"       # 圖形
)
to_install <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(to_install) > 0) install.packages(to_install)
invisible(lapply(pkgs, library, character.only = TRUE))


# ============================================================
# 1) 路徑設定與資料夾建立
# ============================================================

# 原始資料路徑（monthly_panel_final.csv）
IN_PATH <- "C:/我的/政大/碩二下/論文輸出資料與模型/monthly_panel_final.csv"

# 所有 run 的根目錄（資料夾A）
BASE_OUT_DIR <- "C:/我的/政大/碩二下/論文輸出資料與模型/pipeline_runs"

# 本次 run 的時間戳記
RUN_TS <- format(Sys.time(), "%Y%m%d_%H%M%S")

# 本次 run 的資料夾（資料夾B）
OUT_DIR <- file.path(BASE_OUT_DIR, paste0("run_", RUN_TS))

# 子資料夾：pipeline 主要輸出
OUT_DIR_REFIT <- file.path(OUT_DIR, "annual_refit")

# 子資料夾：PCA 對照實驗輸出
OUT_DIR_PCA <- file.path(OUT_DIR, "pca_comparison")

# 子資料夾：論文所需圖表（資料夾B 內的論文專用資料夾）
OUT_DIR_THESIS <- file.path(OUT_DIR, "thesis_output")

# 建立所有資料夾
for (d in c(OUT_DIR, OUT_DIR_REFIT, OUT_DIR_PCA, OUT_DIR_THESIS)) {
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
}

# 啟動 log 檔
logfile <- file.path(OUT_DIR, paste0("log_", RUN_TS, ".txt"))
on.exit(try(sink(), silent = TRUE), add = TRUE)
sink(logfile, split = TRUE)

cat("========================================\n")
cat(" pipeline_main.R | ANNUAL REFITTING (NO LOOK-AHEAD) | ED ONLY\n")
cat("========================================\n")
cat("RUN_TS  :", RUN_TS,    "\n")
cat("IN_PATH :", IN_PATH,   "\n")
cat("OUT_DIR :", OUT_DIR,   "\n\n")


# ============================================================
# 2) 參數設定
# ============================================================

# --- Universe 篩選參數 ---
MCAP_PCT  <- 0.30   # 市值百分位門檻（月末市值百分位 ≥ 30%）
TURN_PCT  <- 0.30   # 流動性百分位門檻（月均成交值百分位 ≥ 30%）
MIN_UNIV  <- 30     # 每月最低股票數

# --- 截面處理參數 ---
MIN_CS  <- 30           # 截面最少有效樣本數
WINS_P  <- c(0.01, 0.99)  # 縮尾上下界

# --- 缺失值門檻（對應論文 3.1 節） ---
RAW_NA_CUT  <- 0.50   # Hard Drop：訓練集原始缺失率上限
POST_NA_CUT <- 0.20   # Soft Drop：補值後殘留缺失率上限

# --- ED 因子數選擇參數（對應論文 3.2.1 節） ---
R_MAX_ED       <- 12    # 最大候選因子數 kmax
ED_TAIL        <- 0.15  # 尾端雜訊估計比例
GAP_MULTIPLIER <- 2     # 雜訊門檻倍數（δ = 2 × d̄）

# --- SOFAR 硬閾值參數（對應論文 3.2.3 節） ---
MIN_NK_PER_FACTOR <- 3      # 每因子最低非零載重數
TAU_SHRINK        <- 0.7    # 保底機制收縮係數
TAU_MIN           <- 0.01   # 閾值下限
FINAL_METHOD      <- "B"    # 閾值方法（c = 0.30/√log p）

# --- 符號校正參數（對應論文 3.2.4 節） ---
SIGN_METHOD        <- "ls_mean"   # 主要方法：多空平均報酬
MIN_MONTHS_SIGN    <- 60          # 最低訓練月份數（避免小樣本偏誤）
SIGN_METHOD_ROBUST <- "ls_tstat"  # 穩健性對照：t 統計量

# --- 回測參數（對應論文 3.3.4 節） ---
NW_LAG       <- 6       # Newey-West 落後期數
TRADING_COST <- 0.006   # 來回交易成本 κ
N_BUCKET     <- 10      # 分位數組數
DO_WINS_RET  <- TRUE    # 是否對報酬縮尾
RET_WINS_P   <- c(0.01, 0.99)  # 報酬縮尾上下界

# --- 樣本期間設定（對應論文 4.1 節） ---
TRAIN_START    <- as.Date("2005-01-01")  # 訓練起始月份
FIRST_OOS_YEAR <- 2016   # 第一個樣本外年份
LAST_OOS_YEAR  <- 2025   # 最後一個樣本外年份

# --- 市場面特徵參數（對應論文 3.3.1 節） ---
RET_VOL_K <- 12   # 報酬波動度滾動窗格（月數）

# --- 機器學習超參數（對應論文附錄 C.1） ---
EN_ALPHA    <- 0.5   # Elastic Net L1/L2 混合比例
RF_TREES    <- 500   # Random Forest 決策樹棵數
RF_MIN_NODE <- 20    # Random Forest 最小節點樣本數

# --- 可重現性種子（對應論文附錄 C.1） ---
GLOBAL_SEED <- 42


# ============================================================
# 3) 讀資料 + QC
# ============================================================

df <- fread(IN_PATH); setDT(df)

# 清理 stock_id 前後空白，ym 統一為月初（YYYY-MM-01）
df[, stock_id := trimws(as.character(stock_id))]
df[, ym := floor_date(as.Date(ym), "month")]

# 確認必要欄位存在
stopifnot(all(c("stock_id", "ym") %in% names(df)))
stopifnot(all(c("mktcap_eom", "avg_turnover_m", "monthly_ret") %in% names(df)))

# 數值型別轉換
df[, monthly_ret    := as.numeric(monthly_ret)]
df[, mktcap_eom     := as.numeric(mktcap_eom)]
df[, avg_turnover_m := as.numeric(avg_turnover_m)]

# 基本維度報告
cat("資料維度：",  nrow(df), "×", ncol(df), "\n")
cat("股票數：",    uniqueN(df$stock_id), "\n")
cat("月份數：",    uniqueN(df$ym), "\n")
cat("時間範圍：",  as.character(min(df$ym, na.rm = TRUE)),
    "~", as.character(max(df$ym, na.rm = TRUE)), "\n\n")

# QC：檢查重複 key
dup_key <- df[, .N, by = .(stock_id, ym)][N > 1]
cat("【QC】重複 key 筆數：", nrow(dup_key), "\n")
if (nrow(dup_key) > 0) {
  print(dup_key[1:min(20, .N)])
  stop("請先處理重複 key 再往下。")
}

# QC：確認 ym 全為月初
ym_day_tab <- df[, .N, by = .(day = as.integer(format(ym, "%d")))][order(day)]
cat("\n【QC】ym day-of-month distribution:\n"); print(ym_day_tab)
if (!all(unique(df[, as.integer(format(ym, "%d"))]) == 1L)) {
  stop("ym 不是月初（YYYY-MM-01），請先執行 floor_date。")
}


# ============================================================
# 4) 市場面特徵計算（對應論文 3.3.1 節）
# ============================================================

setorder(df, stock_id, ym)

# 流動性：月均成交值取自然對數
df[, log_turnover := {
  x <- avg_turnover_m; eps <- 1e-12
  ifelse(is.finite(x) & x > 0, log(pmax(x, eps)), NA_real_)
}]

# 報酬波動度：過去 12 個月月報酬率標準差，至少需 6 個有效觀測值
df[, ret_vol := {
  x <- as.numeric(monthly_ret)
  data.table::frollapply(
    x, n = RET_VOL_K,
    FUN = function(v) {
      v <- as.numeric(v); v <- v[is.finite(v)]
      if (length(v) < max(6, floor(RET_VOL_K * 0.5))) return(NA_real_)
      sd(v)
    },
    align = "right", fill = NA_real_
  )
}, by = stock_id]

# 次月報酬（預測目標變數）
df[, ret_fwd1 := shift(monthly_ret, type = "lead"), by = stock_id]

# 對數報酬輔助函數（用於動能計算）
safe_lr <- function(r) {
  out <- rep(NA_real_, length(r)); ok <- is.finite(r)
  out[ok] <- log1p(r[ok]); out
}

# 動能計算：skip-1 設計（對應論文 3.3.1 節）
df[, lr      := safe_lr(monthly_ret)]
df[, lr_lag1 := shift(lr, 1), by = stock_id]
df[, lr_sum6  := data.table::frollsum(lr_lag1, n = 6,  align = "right",
                                       fill = NA_real_, na.rm = FALSE), by = stock_id]
df[, lr_sum12 := data.table::frollsum(lr_lag1, n = 12, align = "right",
                                       fill = NA_real_, na.rm = FALSE), by = stock_id]

df[, mom_1m  := shift(monthly_ret, 1), by = stock_id]          # 過去 1 個月報酬（無 skip）
df[, mom_6m  := exp(lr_sum6  - lr_lag1) - 1]                   # 6 個月動能（skip-1）
df[, mom_12m := exp(lr_sum12 - lr_lag1) - 1]                   # 12 個月動能（skip-1）
df[, c("lr", "lr_lag1", "lr_sum6", "lr_sum12") := NULL]        # 清除中間欄位

cat("【ADD】ret_vol NA 率：",     round(100 * mean(is.na(df$ret_vol)),     2), "%\n")
cat("【ADD】log_turnover NA 率：", round(100 * mean(is.na(df$log_turnover)), 2), "%\n\n")


# ============================================================
# 5) Universe flag（對應論文 4.2 節）
# ============================================================

df[, universe := {
  mcap <- as.numeric(mktcap_eom); turn <- as.numeric(avg_turnover_m)
  n_m <- sum(is.finite(mcap));    n_t  <- sum(is.finite(turn))
  if (n_m < MIN_UNIV || n_t < MIN_UNIV) {
    rep(FALSE, .N)
  } else {
    mcap_cut <- quantile(mcap, MCAP_PCT, na.rm = TRUE, type = 7)
    turn_cut <- quantile(turn, TURN_PCT, na.rm = TRUE, type = 7)
    is.finite(mcap) & is.finite(turn) & (mcap >= mcap_cut) & (turn >= turn_cut)
  }
}, by = ym]

cat("Universe overall ratio:", round(100 * mean(df$universe, na.rm = TRUE), 2), "%\n\n")


# ============================================================
# 6) 表4 財務特徵描述統計（對應論文 4.3.4 節）
#    ★ 獨立計算：universe==TRUE 原始資料，補值前
#    ★ 對各變數做 1%/99% 縮尾後計算，na.rm=TRUE
# ============================================================

x_vars_all <- c(
  "A_roa", "A_roe", "A_gpm", "A_opm", "A_npm",
  "B_rev_yoy", "B_eps", "B_gp_yoy", "B_op_yoy", "B_np_yoy",
  "C_ocf", "C_fcf",
  "D_cr", "D_qr", "D_dr", "D_icr",
  "E_itr", "E_art", "E_tat",
  "F_ta", "F_te", "F_ca",
  "G_cfi", "G_cff", "G_ebt"
)

# 確認所有財務變數存在
missing_vars <- x_vars_all[!x_vars_all %in% names(df)]
if (length(missing_vars) > 0) {
  cat("❌ 以下欄位不存在：\n"); print(missing_vars)
  stop("請確認欄位名稱。")
}

cat("SOFAR candidate vars:", length(x_vars_all), "\n")

# 取 universe==TRUE 原始資料（補值前）
df_desc <- df[universe == TRUE, c("stock_id", "ym", x_vars_all), with = FALSE]

# 各變數 1%/99% 縮尾後計算描述統計
desc_stats <- rbindlist(lapply(x_vars_all, function(v) {
  x <- as.numeric(df_desc[[v]])
  ok <- is.finite(x)
  n_ok <- sum(ok)
  if (n_ok < MIN_CS) return(NULL)

  # 暫時縮尾（只用於計算統計量，不修改原始資料）
  q <- quantile(x[ok], c(0.01, 0.99), na.rm = TRUE, type = 7)
  x_w <- pmin(pmax(x[ok], q[1]), q[2])

  data.table(
    variable  = v,
    n         = n_ok,
    mean      = mean(x_w,   na.rm = TRUE),
    median    = median(x_w, na.rm = TRUE),
    sd        = sd(x_w,     na.rm = TRUE),
    min       = min(x_w,    na.rm = TRUE),
    max       = max(x_w,    na.rm = TRUE),
    na_rate   = round(mean(!is.finite(as.numeric(df_desc[[v]]))), 4)  # 誠實缺失率
  )
}))

cat("\n【表4】財務特徵描述統計（25項）：\n")
print(desc_stats)

# 輸出至 pipeline run 資料夾
fwrite(desc_stats,
       file.path(OUT_DIR_REFIT, paste0("table4_desc_stats_", RUN_TS, ".csv")))

cat("表4 已儲存。\n\n")


# ============================================================
# 7) 表5 市場面特徵相關係數矩陣（對應論文 4.3.4 節）
#    ★ 獨立計算：universe==TRUE 全樣本
# ============================================================

market_vars_desc <- c("mom_1m", "mom_6m", "mom_12m", "ret_vol", "log_turnover")

# 取 universe==TRUE 並衍生規模與帳面市值比
dt_market <- df[universe == TRUE, .(
  mom_1m, mom_6m, mom_12m, ret_vol, log_turnover,
  size = log(mktcap_eom),
  bm   = ifelse(is.finite(pb_tej) & pb_tej != 0, 1 / pb_tej, NA_real_)
)]
dt_market <- na.omit(dt_market)

cor_mat <- cor(dt_market, use = "complete.obs")
colnames(cor_mat) <- rownames(cor_mat) <- c(
  "Mom1M", "Mom6M", "Mom12M", "波動度", "流動性", "規模", "帳面市值比"
)

cat("\n【表5】市場面特徵相關係數矩陣：\n")
print(round(cor_mat, 3))

# 輸出
cor_dt <- as.data.table(cor_mat, keep.rownames = "variable")
fwrite(cor_dt,
       file.path(OUT_DIR_REFIT, paste0("table5_market_corr_", RUN_TS, ".csv")))

cat("表5 已儲存。\n\n")


# ============================================================
# 8) 圖1 Universe 規模走勢（對應論文 4.2 節）
# ============================================================

univ_size <- df[universe == TRUE, .N, by = ym]
setorder(univ_size, ym)

# 輸出 Universe 規模 CSV（供 output_tables_figures.R 使用）
fwrite(univ_size,
       file.path(OUT_DIR_REFIT, paste0("universe_size_monthly_", RUN_TS, ".csv")))

# 畫圖並存至 pipeline run 資料夾
p_univ <- ggplot(univ_size, aes(x = ym, y = N)) +
  geom_line(color = "#2C3E50", linewidth = 0.7) +
  geom_smooth(method = "loess", span = 0.15,
              color = "#E74C3C", linewidth = 0.8, se = FALSE) +
  scale_y_continuous(limits = c(600, 1200), breaks = seq(600, 1200, 100)) +
  scale_x_date(date_breaks = "2 years", date_labels = "%Y") +
  labs(x = "年份", y = "股票數（檔）") +
  theme_classic() +
  theme(axis.title = element_text(size = 12),
        axis.text  = element_text(size = 11))

ggsave(file.path(OUT_DIR_REFIT, paste0("figure1_universe_size_", RUN_TS, ".png")),
       plot = p_univ, width = 8, height = 5, dpi = 300)

cat("圖1 已儲存。\n\n")


# ============================================================
# 9) Helper Functions
# ============================================================

# --- 截面縮尾（用於特徵前處理）---
winsorize <- function(x) {
  x <- as.numeric(x); ok <- is.finite(x)
  if (sum(ok) < MIN_CS) return(x)
  q <- quantile(x[ok], WINS_P, na.rm = TRUE, type = 7)
  x[ok] <- pmin(pmax(x[ok], q[1]), q[2]); x
}

# --- 截面 Z-score 標準化（用於特徵前處理）---
cs_z <- function(x) {
  x <- as.numeric(x); ok <- is.finite(x)
  if (sum(ok) < MIN_CS) return(x)
  s <- sd(x[ok], na.rm = TRUE)
  if (!is.finite(s) || s == 0) return(x)
  m <- mean(x[ok], na.rm = TRUE); x[ok] <- (x[ok] - m) / s; x
}

# --- 截面 Z-score（回傳 NA 而非原值，用於因子分數標準化）---
cs_z_score <- function(x) {
  x <- as.numeric(x); ok <- is.finite(x)
  if (sum(ok) < 30) return(rep(NA_real_, length(x)))
  s <- sd(x[ok], na.rm = TRUE)
  if (!is.finite(s) || s == 0) return(rep(NA_real_, length(x)))
  m <- mean(x[ok], na.rm = TRUE)
  out <- rep(NA_real_, length(x)); out[ok] <- (x[ok] - m) / s; out
}

# --- 報酬縮尾（用於目標變數前處理）---
winsor_cs_ret <- function(x, p = c(0.01, 0.99)) {
  x <- as.numeric(x); ok <- is.finite(x)
  if (sum(ok) < 100) return(x)
  q <- quantile(x[ok], p, na.rm = TRUE, type = 7)
  pmin(pmax(x, q[1]), q[2])
}

# --- 截面縮尾 v2（用於市場面特徵，對小樣本更寬鬆）---
winsor_cs2 <- function(x, p = c(0.01, 0.99)) {
  x <- as.numeric(x); ok <- is.finite(x)
  if (sum(ok) < 50) return(x)
  q <- suppressWarnings(quantile(x[ok], probs = p, na.rm = TRUE, type = 7))
  if (length(q) < 2 || any(!is.finite(q))) return(x)
  out <- x; out[ok] <- pmin(pmax(x[ok], q[1]), q[2]); out
}

# --- 十分位排序（用於建構多空投資組合）---
assign_bucket <- function(x, n_bucket = 10, sd_min = 1e-8) {
  x <- as.numeric(x); ok <- is.finite(x)
  if (sum(ok) < n_bucket * 5) return(rep(NA_integer_, length(x)))
  sx <- sd(x[ok], na.rm = TRUE)
  if (!is.finite(sx) || sx < sd_min) return(rep(NA_integer_, length(x)))
  qs <- unique(quantile(x[ok], probs = seq(0, 1, length.out = n_bucket + 1),
                        na.rm = TRUE, type = 7))
  if (length(qs) < 3) return(rep(NA_integer_, length(x)))
  as.integer(cut(x, breaks = qs, include.lowest = TRUE, labels = FALSE))
}

# --- 投資組合績效統計（年化報酬、Sharpe、NW t、最大回撤）---
perf_stats <- function(r, freq = 12, nw_lag = NW_LAG) {
  r <- as.numeric(r); r <- r[is.finite(r)]
  if (length(r) < 6) return(list(n = length(r), ann_ret = NA, ann_vol = NA,
                                  sharpe = NA, tstat = NA, tstat_nw = NA,
                                  max_dd = NA))
  mu  <- mean(r); sdv <- sd(r)
  ann_ret <- prod(1 + r)^(freq / length(r)) - 1
  ann_vol <- sdv * sqrt(freq)
  sharpe  <- ifelse(ann_vol > 0, (mu * freq) / ann_vol, NA_real_)
  tstat   <- ifelse(sdv > 0, mu / (sdv / sqrt(length(r))), NA_real_)
  tstat_nw <- tryCatch({
    df_nw <- data.frame(y = r); fit_nw <- lm(y ~ 1, data = df_nw)
    nw_se <- sqrt(as.numeric(
      sandwich::NeweyWest(fit_nw, lag = nw_lag, prewhite = FALSE)[1, 1]))
    ifelse(nw_se > 0, mu / nw_se, NA_real_)
  }, error = function(e) NA_real_)
  cum_ret <- cumprod(1 + r); running_max <- cummax(cum_ret)
  dd <- (cum_ret - running_max) / running_max
  max_dd <- suppressWarnings(min(dd, na.rm = TRUE))
  list(n = length(r), ann_ret = ann_ret, ann_vol = ann_vol,
       sharpe = sharpe, tstat = tstat, tstat_nw = tstat_nw, max_dd = max_dd)
}

# --- 安全讀取欄位（欄位不存在則回傳 NA 向量）---
.safe_pick_col <- function(dt, colname) {
  if (colname %in% names(dt)) dt[[colname]] else rep(NA_real_, nrow(dt))
}

# --- 等權分配（用於換手率計算）---
.make_equal_weight <- function(dt_month) {
  if (nrow(dt_month) == 0) return(data.table())
  dt_month[, n := .N, by = bucket]; dt_month[, w := 1 / n]
  dt_month[, .(stock_id, bucket, w)]
}

# --- 單邊換手率計算（τ = ½ Σ|w_now - w_prev|）---
.calc_turnover_one_side <- function(w_now, w_prev) {
  if (nrow(w_now) == 0 || nrow(w_prev) == 0) return(0)
  d <- merge(w_now, w_prev, by = "stock_id", all = TRUE,
             suffixes = c("_now", "_prev"))
  d[is.na(w_now),  w_now  := 0]
  d[is.na(w_prev), w_prev := 0]
  0.5 * sum(abs(d$w_now - d$w_prev))
}

# --- 預測值排序回測（含交易成本計算，對應論文 3.3.4 節）---
run_pred_sort_bt <- function(dt, score_col, n_bucket = 10,
                              trading_cost = TRADING_COST) {
  dt <- copy(dt)
  dt[, bucket := assign_bucket(get(score_col), n_bucket = n_bucket), by = ym]
  gret <- dt[!is.na(bucket), .(ret = mean(y, na.rm = TRUE), n = .N),
             by = .(ym, bucket)]
  wide <- dcast(gret, ym ~ bucket, value.var = "ret"); setorder(wide, ym)
  top <- as.character(n_bucket); bot <- "1"
  wide[, long_gross  := .safe_pick_col(wide, top)]
  wide[, short_gross := .safe_pick_col(wide, bot)]
  wide[, ls_gross    := long_gross - short_gross]

  # 計算換手率與淨報酬
  months <- sort(unique(dt$ym))
  prev_long_w <- data.table(); prev_short_w <- data.table()
  tc_long <- numeric(length(months)); tc_short <- numeric(length(months))
  for (i in seq_along(months)) {
    m  <- months[i]
    dm <- dt[ym == m & !is.na(bucket), .(stock_id, bucket)]
    if (nrow(dm) == 0) { tc_long[i] <- 0; tc_short[i] <- 0; next }
    wtab    <- .make_equal_weight(dm)
    w_long  <- wtab[bucket == as.integer(top), .(stock_id, w)]
    w_short <- wtab[bucket == as.integer(bot),  .(stock_id, w)]
    tc_long[i]   <- .calc_turnover_one_side(w_long,  prev_long_w)
    tc_short[i]  <- .calc_turnover_one_side(w_short, prev_short_w)
    prev_long_w  <- w_long; prev_short_w <- w_short
  }
  tc_dt <- data.table(ym = as.Date(months),
                      turnover_long  = as.numeric(tc_long),
                      turnover_short = as.numeric(tc_short))
  wide[, ym := as.Date(ym)]
  wide <- merge(wide, tc_dt, by = "ym", all.x = TRUE)
  wide[is.na(turnover_long),  turnover_long  := 0]
  wide[is.na(turnover_short), turnover_short := 0]
  wide[, turnover_ls  := turnover_long + turnover_short]
  wide[, long_net     := long_gross  - turnover_long  * trading_cost]
  wide[, short_net    := short_gross - turnover_short * trading_cost]
  wide[, ls_net       := long_net - short_net]
  wide[, wealth_ls_gross := cumprod(1 + fifelse(is.finite(ls_gross), ls_gross, 0))]
  wide[, wealth_ls_net   := cumprod(1 + fifelse(is.finite(ls_net),   ls_net,   0))]
  list(wide = wide, gret = gret)
}

# --- ED 因子數選擇（對應論文 3.2.1 節）---
ed_select_r <- function(lambda, r_max = 12, tail = 0.15, gap_mult = 2) {
  p <- length(lambda); r_max <- min(r_max, p - 2)
  tail_n <- max(10, floor(p * tail)); idx_tail_start <- p - tail_n + 1
  d_tail <- diff(lambda[idx_tail_start:p])
  edge_hat <- mean(abs(d_tail), na.rm = TRUE)
  gaps <- lambda[1:r_max] - lambda[2:(r_max + 1)]
  gap_threshold <- gap_mult * edge_hat
  candidate_r <- which(gaps > gap_threshold)
  r_hat <- if (length(candidate_r) == 0) 0 else max(candidate_r)
  list(r_hat = r_hat, edge_hat = edge_hat,
       gaps = gaps, gap_threshold = gap_threshold)
}

# --- 硬閾值保底機制（對應論文 3.2.3 節）---
apply_threshold_with_guard <- function(U, tau, min_nk, shrink, tau_min) {
  U_thr <- U; U_thr[abs(U_thr) < tau] <- 0
  Nk <- colSums(U_thr != 0); iter <- 0L
  while (any(Nk < min_nk) && tau > tau_min && iter < 20L) {
    tau <- tau * shrink; U_thr <- U; U_thr[abs(U_thr) < tau] <- 0
    Nk <- colSums(U_thr != 0); iter <- iter + 1L
  }
  list(U_thr = U_thr, tau = tau, Nk = Nk, iter = iter)
}

# --- 閾值選擇（c = 0.30/√log p，對應 FINAL_METHOD = "B"）---
pick_thr <- function(U_raw, p_var, method) {
  tau0 <- switch(method,
                 "A" = 0.20 / sqrt(log(p_var)),
                 "B" = 0.30 / sqrt(log(p_var)),
                 "C" = 0.35 / sqrt(log(p_var)),
                 "D" = 0.40 / sqrt(log(p_var)),
                 "E" = as.numeric(quantile(abs(U_raw), 0.20, na.rm = TRUE)),
                 stop("FINAL_METHOD must be A/B/C/D/E"))
  thr <- apply_threshold_with_guard(U_raw, tau0,
                                    MIN_NK_PER_FACTOR, TAU_SHRINK, TAU_MIN)
  list(thr = thr, tau = thr$tau,
       desc = paste0("method=", method,
                     " tau0=", round(tau0, 4),
                     " tau_final=", round(thr$tau, 4),
                     " iter=", thr$iter,
                     " minNk=", min(thr$Nk),
                     " avgNk=", round(mean(thr$Nk), 2)))
}

# --- 建構特徵矩陣（NA 補零）---
make_X <- function(d, cols) {
  if (nrow(d) == 0 || length(cols) == 0) return(NULL)
  X <- as.matrix(d[, ..cols]); X[!is.finite(X)] <- 0; X
}

# --- XGBoost 訓練並預測（固定超參數，對應論文附錄 C.1）---
.xgb_fit_predict <- function(X_tr, y_tr, X_te) {
  dtr <- xgboost::xgb.DMatrix(data = X_tr, label = y_tr)
  dte <- xgboost::xgb.DMatrix(data = X_te)
  params <- list(
    objective        = "reg:squarederror",
    eval_metric      = "rmse",
    eta              = 0.05,     # 學習率
    max_depth        = 4,        # 最大樹深
    subsample        = 0.8,      # 樣本抽樣比例
    colsample_bytree = 0.8,      # 特徵抽樣比例
    seed             = GLOBAL_SEED
  )
  bst <- xgboost::xgb.train(params = params, data = dtr,
                             nrounds = 300, verbose = 0)
  as.numeric(predict(bst, dte))
}

# --- SOFAR 安全包裝（修正 rrpack ic.type 長度非 1 之 bug）---
sofar_safe <- function(Y, X, nrank, ic_type_prefer = "BIC", ...) {
  f <- rrpack::sofar; fn <- names(formals(f)); dots <- list(...)
  if ("ic.type" %in% fn) {
    if (is.null(dots$ic.type) || length(dots$ic.type) != 1L ||
        !is.character(dots$ic.type))
      dots$ic.type <- ic_type_prefer
    else dots$ic.type <- as.character(dots$ic.type)[1]
  }
  if ("ictype" %in% fn) {
    if (is.null(dots$ictype) || length(dots$ictype) != 1L ||
        !is.character(dots$ictype))
      dots$ictype <- ic_type_prefer
    else dots$ictype <- as.character(dots$ictype)[1]
  }
  do.call(f, c(list(Y = Y, X = X, nrank = nrank), dots))
}

# --- 純因子排序回測（訓練期符號校正用）---
run_factor_bt_simple <- function(dt, score_col, n_bucket = 10, ret_col = NULL) {
  dt <- copy(dt)
  if (is.null(ret_col))
    ret_col <- if ("ret_fwd1_w" %in% names(dt)) "ret_fwd1_w" else "ret_fwd1"
  stopifnot(ret_col %in% names(dt))
  dt <- dt[is.finite(get(ret_col))]
  dt[, bucket := assign_bucket(get(score_col), n_bucket = n_bucket), by = ym]
  gret <- dt[!is.na(bucket),
             .(ret = mean(get(ret_col), na.rm = TRUE)), by = .(ym, bucket)]
  wide <- dcast(gret, ym ~ bucket, value.var = "ret"); setorder(wide, ym)
  top <- as.character(n_bucket); bot <- "1"
  wide[, long_gross  := .safe_pick_col(wide, top)]
  wide[, short_gross := .safe_pick_col(wide, bot)]
  wide[, ls_gross    := long_gross - short_gross]
  wide
}


# ============================================================
# 10) NA Tracking：訓練期擴展窗格（2005–2015）
#     對應論文 4.3.3 節：補值流程缺失率趨勢
#     設計原則：與 OOS loop 定義完全一致，可與 OOS 合併為完整趨勢表
# ============================================================

cat("\n========================================\n")
cat("[NA Tracking] 訓練期擴展窗格（2005–2015）\n")
cat("========================================\n")

x_vars_tr_base <- intersect(x_vars_all, names(df))

na_train_byyear <- lapply(2005:2015, function(yr) {
  train_end_yr <- as.Date(paste0(yr, "-12-01"))

  # 擴展窗格：TRAIN_START ~ 該年年底（與 OOS loop 訓練期定義一致）
  dt_exp <- df[universe == TRUE & ym >= TRAIN_START & ym <= train_end_yr,
               c("stock_id", "ym", x_vars_tr_base), with = FALSE]
  if (nrow(dt_exp) == 0) return(NULL)
  x_v <- intersect(x_vars_tr_base, names(dt_exp))

  # Step 1: 原始缺失率
  na_raw <- sapply(x_v, function(v) mean(!is.finite(dt_exp[[v]])))
  r1 <- mean(na_raw)
  x_v_keep <- names(na_raw)[na_raw <= RAW_NA_CUT]
  n_dropped_raw <- length(x_v) - length(x_v_keep)

  # Step 2: LOCF 填補
  dt_exp2 <- copy(dt_exp[, c("stock_id", "ym", x_v_keep), with = FALSE])
  setorder(dt_exp2, stock_id, ym)
  for (v in x_v_keep)
    dt_exp2[, (v) := nafill(get(v), type = "locf"), by = stock_id]
  r2 <- mean(sapply(x_v_keep, function(v) mean(!is.finite(dt_exp2[[v]]))))

  # Step 3: 截面中位數填補
  for (v in x_v_keep) {
    dt_exp2[, tmp_med := {
      x <- as.numeric(get(v)); ok <- is.finite(x)
      if (sum(ok) < MIN_CS) NA_real_ else median(x[ok], na.rm = TRUE)
    }, by = ym]
    dt_exp2[!is.finite(get(v)), (v) := tmp_med]
    dt_exp2[, tmp_med := NULL]
  }
  r3 <- mean(sapply(x_v_keep, function(v) mean(!is.finite(dt_exp2[[v]]))))

  data.table(
    year             = yr,
    n_obs            = nrow(dt_exp),
    n_stocks         = uniqueN(dt_exp$stock_id),
    n_vars           = length(x_v_keep),
    n_dropped_raw    = n_dropped_raw,
    step_1_raw       = round(r1, 6),
    step_2_locf      = round(r2, 6),
    step_3_cs_median = round(r3, 6)
  )
})

na_train_byyear_dt <- rbindlist(Filter(Negate(is.null), na_train_byyear))

fwrite(na_train_byyear_dt,
       file.path(OUT_DIR_REFIT,
                 paste0("na_tracking_TRAIN_expanding_byyear_", RUN_TS, ".csv")))

cat("\n【訓練期逐年 NA rate（Expanding Window）】\n")
print(na_train_byyear_dt[, .(year, n_obs, n_stocks,
                              step_1_raw, step_2_locf, step_3_cs_median)])

cat(sprintf("\n訓練期（2005–2015）step_1_raw 平均：%.1f%%（最高：%.1f%%，最低：%.1f%%）\n",
            100 * mean(na_train_byyear_dt$step_1_raw),
            100 * max(na_train_byyear_dt$step_1_raw),
            100 * min(na_train_byyear_dt$step_1_raw)))
cat(sprintf("訓練期（2005–2015）step_3_cs_median 平均：%.1f%%\n",
            100 * mean(na_train_byyear_dt$step_3_cs_median)))
cat("\n========================================\n\n")


# ============================================================
# 11) Annual Refitting Loop
#     對應論文 3.2 節（SOFAR）+ 3.3 節（ML）+ 3.3.3 節（動態回測框架）
#     擴展窗格：訓練至 y-1 年底，預測 y 年全年
# ============================================================

pred_all    <- list()    # 儲存各年預測值 panel
pred_bt_sum <- data.table()  # 儲存各年各模型逐年績效

years <- FIRST_OOS_YEAR:LAST_OOS_YEAR
cat("OOS years:", paste(years, collapse = ", "), "\n\n")

for (yy in years) {

  # 每年起始固定種子，確保 RF / XGBoost / EN 完全可重現
  set.seed(GLOBAL_SEED)

  cat("====================================================\n")
  cat("ANNUAL REFIT YEAR:", yy, "\n")
  cat("====================================================\n")

  # 設定訓練期與測試期邊界
  train_end  <- as.Date(paste0(yy - 1, "-12-01"))
  test_start <- as.Date(paste0(yy,     "-01-01"))
  test_end   <- as.Date(paste0(yy,     "-12-01"))

  cat("Train window:", as.character(TRAIN_START), "~", as.character(train_end), "\n")
  cat("Test  window:", as.character(test_start),  "~", as.character(test_end),  "\n")

  # 取 universe 內資料
  dsub <- df[universe == TRUE & ym >= TRAIN_START & ym <= test_end]
  dtr0 <- dsub[ym <= train_end]
  dte0 <- dsub[ym >= test_start & ym <= test_end]

  cat("Train rows:", nrow(dtr0), "| Test rows:", nrow(dte0), "\n")
  if (nrow(dtr0) == 0 || nrow(dte0) == 0) {
    cat("⚠️ skip year", yy, "(empty train/test)\n\n"); next
  }

  x_vars <- intersect(x_vars_all, names(dtr0))
  if (length(x_vars) < 5) {
    cat("⚠️ skip year", yy, "(too few x_vars)\n\n"); next
  }

  # -------------------------------------------------------
  # 11A) 資料前處理（對應論文 3.1 節七步驟）
  # -------------------------------------------------------

  # 合併訓練集與測試集做統一前處理（LOCF 需完整序列）
  dt_u <- rbind(
    dtr0[, c("stock_id", "ym", x_vars), with = FALSE],
    dte0[, c("stock_id", "ym", x_vars), with = FALSE],
    fill = TRUE
  )
  setorder(dt_u, stock_id, ym)
  na_track <- list()   # 儲存各步驟缺失率

  # Step 1: Hard Drop（訓練集原始缺失率 > RAW_NA_CUT 之變數剔除）
  dt_tr_raw   <- dt_u[ym <= train_end]
  na_rate_raw <- sapply(x_vars, function(v) mean(!is.finite(dt_tr_raw[[v]])))
  drop_raw    <- names(na_rate_raw)[na_rate_raw > RAW_NA_CUT]
  if (length(drop_raw) > 0) {
    cat("Hard Drop vars:", length(drop_raw), "\n")
    x_vars <- setdiff(x_vars, drop_raw)
    dt_u   <- dt_u[, c("stock_id", "ym", x_vars), with = FALSE]
  }
  if (length(x_vars) < 5) {
    cat("⚠️ skip year", yy, "(too few vars after Hard Drop)\n\n"); next
  }
  na_rate_step1 <- mean(sapply(x_vars, function(v) mean(!is.finite(dt_u[[v]]))))
  na_track[["1_raw_drop"]] <- data.table(year = yy, step = "1_raw_drop",
                                          na_rate = na_rate_step1,
                                          n_vars  = length(x_vars))
  cat("NA rate after Hard Drop:", round(100 * na_rate_step1, 2), "%\n")

  # Step 2: LOCF 時間序列向前填補
  for (v in x_vars) dt_u[, (v) := nafill(get(v), type = "locf"), by = stock_id]
  na_rate_step2 <- mean(sapply(x_vars, function(v) mean(!is.finite(dt_u[[v]]))))
  na_track[["2_locf"]] <- data.table(year = yy, step = "2_locf",
                                      na_rate = na_rate_step2,
                                      n_vars  = length(x_vars))
  cat("NA rate after LOCF:", round(100 * na_rate_step2, 2), "%\n")

  # Step 3: 截面中位數填補（新上市公司無歷史資料）
  for (v in x_vars) {
    dt_u[, tmp_med := {
      x <- as.numeric(get(v)); ok <- is.finite(x)
      if (sum(ok) < MIN_CS) NA_real_ else median(x[ok], na.rm = TRUE)
    }, by = ym]
    dt_u[!is.finite(get(v)), (v) := tmp_med]
    dt_u[, tmp_med := NULL]
  }
  na_rate_step3 <- mean(sapply(x_vars, function(v) mean(!is.finite(dt_u[[v]]))))
  na_track[["3_cs_median"]] <- data.table(year = yy, step = "3_cs_median",
                                           na_rate = na_rate_step3,
                                           n_vars  = length(x_vars))
  cat("NA rate after CS Median:", round(100 * na_rate_step3, 2), "%\n")

  # Step 4 & 5: 截面縮尾 + Z-score 標準化
  for (v in x_vars) dt_u[, (v) := winsorize(get(v)), by = ym]
  for (v in x_vars) dt_u[, (v) := cs_z(get(v)),     by = ym]

  # Step 6: Soft Drop（補值後殘留缺失率 > POST_NA_CUT 之變數剔除）
  dt_tr_post  <- dt_u[ym <= train_end]
  na_rate_post <- sapply(x_vars, function(v) mean(!is.finite(dt_tr_post[[v]])))
  drop_post   <- names(na_rate_post)[na_rate_post > POST_NA_CUT]
  if (length(drop_post) > 0) {
    cat("Soft Drop vars:", length(drop_post), "\n")
    x_vars <- setdiff(x_vars, drop_post)
    dt_u   <- dt_u[, c("stock_id", "ym", x_vars), with = FALSE]
  }
  if (length(x_vars) < 5) {
    cat("⚠️ skip year", yy, "(too few vars after Soft Drop)\n\n"); next
  }
  na_rate_step6 <- mean(sapply(x_vars, function(v) mean(!is.finite(dt_u[[v]]))))
  na_track[["6_post_drop"]] <- data.table(year = yy, step = "6_post_drop",
                                           na_rate = na_rate_step6,
                                           n_vars  = length(x_vars))
  cat("NA rate after Soft Drop:", round(100 * na_rate_step6, 2), "%\n")

  # Step 7: 補零（SOFAR 需無缺失輸入矩陣）
  for (v in x_vars) dt_u[!is.finite(get(v)), (v) := 0]
  na_rate_step7 <- 0   # 補零後必為 0
  na_track[["7_fill0"]] <- data.table(year = yy, step = "7_fill0",
                                       na_rate = 0, n_vars = length(x_vars))

  # 拆回訓練集與測試集
  dt_tr <- dt_u[ym <= train_end]
  dt_te <- dt_u[ym >= test_start & ym <= test_end]

  X_tr <- as.matrix(dt_tr[, ..x_vars]); X_tr[!is.finite(X_tr)] <- 0
  X_te <- as.matrix(dt_te[, ..x_vars]); X_te[!is.finite(X_te)] <- 0

  cat("SOFAR X_tr dim:", nrow(X_tr), "×", ncol(X_tr), "\n")

  # 儲存 drop log
  fwrite(data.table(year          = yy,
                    raw_drop_n    = length(drop_raw),
                    post_drop_n   = length(drop_post),
                    x_vars_final  = length(x_vars),
                    raw_drop_vars = paste(drop_raw,  collapse = " | "),
                    post_drop_vars= paste(drop_post, collapse = " | ")),
         file.path(OUT_DIR_REFIT, paste0("drop_log_year_", yy, ".csv")))

  # -------------------------------------------------------
  # 11B) ED 因子數選擇（對應論文 3.2.1 節）
  # -------------------------------------------------------
  sv     <- svd(X_tr, nu = 0, nv = 0)
  lambda <- sort((sv$d^2) / (nrow(X_tr) - 1), decreasing = TRUE)
  ed     <- ed_select_r(lambda, r_max = R_MAX_ED,
                        tail = ED_TAIL, gap_mult = GAP_MULTIPLIER)
  best_k <- ed$r_hat
  if (best_k < 1) { cat("⚠️ ED picked 0 → force 1\n"); best_k <- 1L }
  cat("ED best_k:", best_k, "| edge_hat:", ed$edge_hat,
      "| threshold:", ed$gap_threshold, "\n")

  fwrite(data.table(year = yy, method = "ED", nrank = best_k,
                    edge_hat = ed$edge_hat, gap_threshold = ed$gap_threshold,
                    r_max = R_MAX_ED, tail = ED_TAIL, gap_mult = GAP_MULTIPLIER),
         file.path(OUT_DIR_REFIT, paste0("rank_ED_year_", yy, ".csv")))

  # -------------------------------------------------------
  # 11C) SOFAR 估計（對應論文 3.2.2 節）
  # -------------------------------------------------------
  cat("Fit SOFAR on TRAIN only...\n")
  t0  <- Sys.time()
  fit <- tryCatch(
    sofar_safe(Y = X_tr, X = X_tr, nrank = best_k, ic_type_prefer = "BIC"),
    error = function(e) { cat("❌ sofar failed:", conditionMessage(e), "\n"); NULL }
  )
  if (is.null(fit)) {
    cat("⚠️ skip year", yy, "(sofar failed)\n\n"); next
  }
  cat("SOFAR seconds:", round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1), "\n")

  U_raw <- as.matrix(fit$U)
  if (is.null(U_raw)) {
    cat("❌ fit$U missing. skip year", yy, "\n\n"); next
  }

  # 動態硬閾值除躁（對應論文 3.2.3 節）
  p_var <- ncol(X_tr)
  sel   <- pick_thr(U_raw, p_var, FINAL_METHOD)
  U_thr <- sel$thr$U_thr
  cat("Threshold:", sel$desc, "\n")

  # 計算因子分數：F = X × U_thr
  F_tr <- X_tr %*% U_thr; F_te <- X_te %*% U_thr
  colnames(F_tr) <- paste0("sofar_f", 1:ncol(F_tr), "_raw")
  colnames(F_te) <- paste0("sofar_f", 1:ncol(F_te), "_raw")

  # 儲存因子載荷
  Uthr_dt <- as.data.table(U_thr)
  setnames(Uthr_dt, paste0("F", 1:ncol(U_thr)))
  Uthr_dt[, var := x_vars]
  fwrite(Uthr_dt,
         file.path(OUT_DIR_REFIT, paste0("sofar_loadings_thr_year_", yy, ".csv")))

  # -------------------------------------------------------
  # 11D) 符號校正（對應論文 3.2.4 節）
  # -------------------------------------------------------

  # 合併目標變數至 panel
  merge_cols <- intersect(
    c("stock_id", "ym", "ret_fwd1", "mktcap_eom", "pb_tej",
      "mom_12m", "mom_1m", "mom_6m", "ret_vol", "log_turnover"),
    names(dtr0)
  )
  tr_y <- dtr0[, ..merge_cols]; te_y <- dte0[, ..merge_cols]

  tr_panel <- cbind(dt_tr[, .(stock_id, ym)], as.data.table(F_tr))
  te_panel <- cbind(dt_te[, .(stock_id, ym)], as.data.table(F_te))
  tr_panel <- merge(tr_panel, tr_y, by = c("stock_id", "ym"), all.x = TRUE)
  te_panel <- merge(te_panel, te_y, by = c("stock_id", "ym"), all.x = TRUE)

  # 目標變數縮尾
  if (DO_WINS_RET) {
    tr_panel[, ret_fwd1_w := winsor_cs_ret(ret_fwd1, p = RET_WINS_P), by = ym]
    te_panel[, ret_fwd1_w := winsor_cs_ret(ret_fwd1, p = RET_WINS_P), by = ym]
    RET_COL <- "ret_fwd1_w"
  } else {
    RET_COL <- "ret_fwd1"
  }

  f_raw_cols <- colnames(F_tr)
  sign_tab   <- data.table(
    factor_raw = f_raw_cols,
    factor_sc  = sub("_raw$", "_sc", f_raw_cols),
    n_months   = NA_integer_, ls_mean = NA_real_,
    ls_tstat   = NA_real_, sign = 1L, note = ""
  )

  # 主要方法：以訓練期多空平均報酬判斷翻轉方向
  for (fc in f_raw_cols) {
    wide_in <- run_factor_bt_simple(tr_panel, score_col = fc,
                                    n_bucket = N_BUCKET, ret_col = RET_COL)
    r_ls <- wide_in$ls_gross; st <- perf_stats(r_ls)
    sign_tab[factor_raw == fc,
             `:=`(n_months = st$n,
                  ls_mean  = mean(r_ls[is.finite(r_ls)], na.rm = TRUE),
                  ls_tstat = st$tstat)]
    if (!is.finite(st$n) || st$n < MIN_MONTHS_SIGN) {
      sign_tab[factor_raw == fc,
               `:=`(sign = 1L,
                    note = paste0("keep (months<", MIN_MONTHS_SIGN, ")"))]
    } else {
      sgn <- if (SIGN_METHOD == "ls_mean") {
        ifelse(is.finite(sign_tab[factor_raw == fc, ls_mean]) &&
               sign_tab[factor_raw == fc, ls_mean] < 0, -1L, 1L)
      } else {
        ifelse(is.finite(sign_tab[factor_raw == fc, ls_tstat]) &&
               sign_tab[factor_raw == fc, ls_tstat] < 0, -1L, 1L)
      }
      sign_tab[factor_raw == fc,
               `:=`(sign = sgn,
                    note = ifelse(sgn == -1L, "FLIP", "keep"))]
    }
  }
  fwrite(sign_tab,
         file.path(OUT_DIR_REFIT,
                   paste0("factor_signs_train_year_", yy, ".csv")))

  # 套用主要符號校正
  for (i in seq_len(nrow(sign_tab))) {
    rawc <- sign_tab$factor_raw[i]; scc <- sign_tab$factor_sc[i]
    sgn  <- sign_tab$sign[i]
    tr_panel[, (scc) := as.numeric(get(rawc)) * sgn]
    te_panel[, (scc) := as.numeric(get(rawc)) * sgn]
  }
  f_sc_cols <- sign_tab$factor_sc

  # 因子分數截面 Z-score 後等權合成（純 SOFAR 複合因子）
  tr_panel[, paste0("z_", f_sc_cols) := lapply(.SD, cs_z_score),
           by = ym, .SDcols = f_sc_cols]
  te_panel[, paste0("z_", f_sc_cols) := lapply(.SD, cs_z_score),
           by = ym, .SDcols = f_sc_cols]
  z_cols <- paste0("z_", f_sc_cols)
  tr_panel[, sofar_composite_sc  := rowMeans(.SD, na.rm = TRUE), .SDcols = z_cols]
  te_panel[, sofar_composite_sc  := rowMeans(.SD, na.rm = TRUE), .SDcols = z_cols]

  # 穩健性對照：以 t 統計量判斷翻轉方向（對應論文 5.5 節）
  sign_tab_rob <- copy(sign_tab); sign_tab_rob[, `:=`(sign = 1L, note = "")]
  for (fc in f_raw_cols) {
    wide_in <- run_factor_bt_simple(tr_panel, score_col = fc,
                                    n_bucket = N_BUCKET, ret_col = RET_COL)
    r_ls <- wide_in$ls_gross; st <- perf_stats(r_ls)
    if (!is.finite(st$n) || st$n < MIN_MONTHS_SIGN) {
      sign_tab_rob[factor_raw == fc,
                   `:=`(sign = 1L,
                        note = paste0("keep (months<", MIN_MONTHS_SIGN, ")"))]
    } else {
      sgn <- ifelse(is.finite(st$tstat) && st$tstat < 0, -1L, 1L)
      sign_tab_rob[factor_raw == fc,
                   `:=`(sign = sgn,
                        note = ifelse(sgn == -1L, "FLIP", "keep"))]
    }
  }
  fwrite(sign_tab_rob,
         file.path(OUT_DIR_REFIT,
                   paste0("factor_signs_train_year_", yy, "_ROBUST_tstat.csv")))

  # 套用穩健性符號校正
  for (i in seq_len(nrow(sign_tab_rob))) {
    rawc <- sign_tab_rob$factor_raw[i]; sgn <- sign_tab_rob$sign[i]
    tr_panel[, paste0(rawc, "_scR") := as.numeric(get(rawc)) * sgn]
    te_panel[, paste0(rawc, "_scR") := as.numeric(get(rawc)) * sgn]
  }
  f_scR_cols <- paste0(f_raw_cols, "_scR")
  tr_panel[, paste0("z_", f_scR_cols) := lapply(.SD, cs_z_score),
           by = ym, .SDcols = f_scR_cols]
  te_panel[, paste0("z_", f_scR_cols) := lapply(.SD, cs_z_score),
           by = ym, .SDcols = f_scR_cols]
  zR_cols <- paste0("z_", f_scR_cols)
  tr_panel[, sofar_composite_scR := rowMeans(.SD, na.rm = TRUE), .SDcols = zR_cols]
  te_panel[, sofar_composite_scR := rowMeans(.SD, na.rm = TRUE), .SDcols = zR_cols]

  # -------------------------------------------------------
  # 11E) 市場面特徵前處理（對應論文 3.3.1 節）
  # -------------------------------------------------------

  tr_panel[, size_raw := if ("mktcap_eom" %in% names(tr_panel))
    log(pmax(as.numeric(mktcap_eom), 1e-6)) else NA_real_]
  te_panel[, size_raw := if ("mktcap_eom" %in% names(te_panel))
    log(pmax(as.numeric(mktcap_eom), 1e-6)) else NA_real_]
  tr_panel[, bm_raw := if ("pb_tej" %in% names(tr_panel))
    1 / as.numeric(pb_tej) else NA_real_]
  te_panel[, bm_raw := if ("pb_tej" %in% names(te_panel))
    1 / as.numeric(pb_tej) else NA_real_]
  tr_panel[, mom_raw := if ("mom_12m" %in% names(tr_panel))
    as.numeric(mom_12m) else NA_real_]
  te_panel[, mom_raw := if ("mom_12m" %in% names(te_panel))
    as.numeric(mom_12m) else NA_real_]

  market_wanted <- c("mom_1m", "mom_6m", "mom_12m", "ret_vol", "log_turnover")
  for (v in market_wanted) {
    if (v %in% names(tr_panel)) tr_panel[, (v) := as.numeric(get(v))]
    if (v %in% names(te_panel)) te_panel[, (v) := as.numeric(get(v))]
  }

  pred_vars_all <- unique(c("size_raw", "bm_raw", "mom_raw", market_wanted))
  pred_vars_all <- pred_vars_all[pred_vars_all %in% names(tr_panel)]
  for (v in pred_vars_all) {
    tr_panel[, (v)          := winsor_cs2(get(v), p = WINS_P), by = ym]
    te_panel[, (v)          := winsor_cs2(get(v), p = WINS_P), by = ym]
    tr_panel[, paste0("z_", v) := cs_z_score(get(v)), by = ym]
    te_panel[, paste0("z_", v) := cs_z_score(get(v)), by = ym]
  }

  market_z  <- intersect(paste0("z_", market_wanted), names(tr_panel))
  ols3_cols <- intersect(c("z_size_raw", "z_bm_raw", "z_mom_raw"), names(tr_panel))
  hybrid_cols <- unique(c(f_sc_cols, market_z))

  cat("Feature counts | SOFAR:", length(f_sc_cols),
      "| Market:", length(market_z),
      "| Hybrid:", length(hybrid_cols),
      "| OLS3:", length(ols3_cols), "\n")

  # -------------------------------------------------------
  # 11F) 機器學習訓練與預測（對應論文 3.3.2 節）
  # -------------------------------------------------------

  tr_ml <- tr_panel[is.finite(get(RET_COL))]
  te_ml <- te_panel[is.finite(get(RET_COL))]
  if (nrow(tr_ml) == 0 || nrow(te_ml) == 0) {
    cat("⚠️ skip year", yy, "(no finite y)\n\n"); next
  }
  y_tr <- as.numeric(tr_ml[[RET_COL]])
  y_te <- as.numeric(te_ml[[RET_COL]])

  # --- SOFAR-only ---
  Xs_tr <- make_X(tr_ml, f_sc_cols); Xs_te <- make_X(te_ml, f_sc_cols)
  set.seed(GLOBAL_SEED)   # 固定 EN CV 折疊切割
  cvfit_s   <- glmnet::cv.glmnet(x = Xs_tr, y = y_tr, alpha = EN_ALPHA,
                                   nfolds = 5, standardize = TRUE)
  pred_en_s <- as.numeric(predict(cvfit_s, newx = Xs_te, s = "lambda.min"))

  RF_MTRY_s <- max(1, floor(sqrt(ncol(Xs_tr))))
  tr_rf_s   <- data.frame(y = y_tr, Xs_tr)
  colnames(tr_rf_s) <- c("y", f_sc_cols)
  rf_fit_s  <- ranger::ranger(y ~ ., data = tr_rf_s,
                               num.trees = RF_TREES, mtry = RF_MTRY_s,
                               min.node.size = RF_MIN_NODE, seed = GLOBAL_SEED)
  pred_rf_s <- as.numeric(predict(rf_fit_s,
                                   data = data.frame(Xs_te))$predictions)
  pred_xgb_s <- .xgb_fit_predict(Xs_tr, y_tr, Xs_te)

  # --- Market-only ---
  pred_en_m <- pred_rf_m <- pred_xgb_m <- rep(NA_real_, nrow(te_ml))
  if (length(market_z) >= 2) {
    Xm_tr <- make_X(tr_ml, market_z); Xm_te <- make_X(te_ml, market_z)
    set.seed(GLOBAL_SEED)
    cvfit_m   <- glmnet::cv.glmnet(x = Xm_tr, y = y_tr, alpha = EN_ALPHA,
                                    nfolds = 5, standardize = TRUE)
    pred_en_m <- as.numeric(predict(cvfit_m, newx = Xm_te, s = "lambda.min"))

    RF_MTRY_m <- max(1, floor(sqrt(ncol(Xm_tr))))
    tr_rf_m   <- data.frame(y = y_tr, Xm_tr)
    colnames(tr_rf_m) <- c("y", market_z)
    rf_fit_m  <- ranger::ranger(y ~ ., data = tr_rf_m,
                                 num.trees = RF_TREES, mtry = RF_MTRY_m,
                                 min.node.size = RF_MIN_NODE, seed = GLOBAL_SEED)
    pred_rf_m  <- as.numeric(predict(rf_fit_m,
                                      data = data.frame(Xm_te))$predictions)
    pred_xgb_m <- .xgb_fit_predict(Xm_tr, y_tr, Xm_te)
  }

  # --- Hybrid ---
  Xh_tr <- make_X(tr_ml, hybrid_cols); Xh_te <- make_X(te_ml, hybrid_cols)
  set.seed(GLOBAL_SEED)
  cvfit_h   <- glmnet::cv.glmnet(x = Xh_tr, y = y_tr, alpha = EN_ALPHA,
                                   nfolds = 5, standardize = TRUE)
  pred_en_h <- as.numeric(predict(cvfit_h, newx = Xh_te, s = "lambda.min"))

  RF_MTRY_h <- max(1, floor(sqrt(ncol(Xh_tr))))
  tr_rf_h   <- data.frame(y = y_tr, Xh_tr)
  colnames(tr_rf_h) <- c("y", hybrid_cols)
  rf_fit_h  <- ranger::ranger(y ~ ., data = tr_rf_h,
                               num.trees = RF_TREES, mtry = RF_MTRY_h,
                               min.node.size = RF_MIN_NODE, seed = GLOBAL_SEED)
  pred_rf_h  <- as.numeric(predict(rf_fit_h,
                                    data = data.frame(Xh_te))$predictions)
  pred_xgb_h <- .xgb_fit_predict(Xh_tr, y_tr, Xh_te)

  # --- OLS-3 基準模型（規模 + 帳面市值比 + 12月動能）---
  pred_ols3 <- rep(NA_real_, nrow(te_ml))
  if (length(ols3_cols) == 3) {
    tr_ols <- tr_ml[, c(RET_COL, ols3_cols), with = FALSE]
    te_ols <- te_ml[, c(RET_COL, ols3_cols), with = FALSE]
    setnames(tr_ols, RET_COL, "y"); setnames(te_ols, RET_COL, "y")
    tr_ols <- tr_ols[is.finite(y)]
    for (cc in ols3_cols) {
      tr_ols[!is.finite(get(cc)), (cc) := 0]
      te_ols[!is.finite(get(cc)), (cc) := 0]
    }
    fml     <- as.formula(paste("y ~", paste(ols3_cols, collapse = " + ")))
    fit_ols <- tryCatch(lm(fml, data = tr_ols), error = function(e) NULL)
    if (!is.null(fit_ols))
      pred_ols3 <- as.numeric(predict(fit_ols, newdata = te_ols))
  }

  # 彙整預測結果
  out <- te_ml[, .(stock_id, ym)]; out[, y := y_te]; out[, year_refit := yy]
  out[, pred_en_S  := pred_en_s];  out[, pred_rf_S  := pred_rf_s]
  out[, pred_xgb_S := pred_xgb_s]
  out[, pred_en_M  := pred_en_m];  out[, pred_rf_M  := pred_rf_m]
  out[, pred_xgb_M := pred_xgb_m]
  out[, pred_en_H  := pred_en_h];  out[, pred_rf_H  := pred_rf_h]
  out[, pred_xgb_H := pred_xgb_h]
  out[, pred_ols3  := pred_ols3]
  out[, pred_sofar_comp_main := te_ml$sofar_composite_sc]
  out[, pred_sofar_comp_rob  := te_ml$sofar_composite_scR]

  pred_all[[as.character(yy)]] <- out

  # 儲存當年特徵矩陣（供特徵重要性計算用）
  if (yy == LAST_OOS_YEAR) {
    Xh_tr_final     <- Xh_tr
    y_tr_final       <- y_tr
    hybrid_cols_final <- hybrid_cols
  }

  # --- 逐年各模型回測績效 ---
  pred_cols <- grep("^pred_", names(out), value = TRUE)
  for (pc in pred_cols) {
    bt   <- run_pred_sort_bt(out, score_col = pc, n_bucket = N_BUCKET,
                              trading_cost = TRADING_COST)
    st_g <- perf_stats(bt$wide$ls_gross)
    st_n <- perf_stats(bt$wide$ls_net)
    pred_bt_sum <- rbind(pred_bt_sum,
                         data.table(
                           year_refit       = yy, model = pc,
                           trading_cost     = TRADING_COST,
                           n_months_gross   = st_g$n,
                           ann_ret_ls_gross = st_g$ann_ret,
                           sharpe_ls_gross  = st_g$sharpe,
                           tstat_ls_gross   = st_g$tstat,
                           tstat_nw_ls_gross= st_g$tstat_nw,
                           maxdd_ls_gross   = st_g$max_dd,
                           n_months_net     = st_n$n,
                           ann_ret_ls_net   = st_n$ann_ret,
                           sharpe_ls_net    = st_n$sharpe,
                           tstat_ls_net     = st_n$tstat,
                           tstat_nw_ls_net  = st_n$tstat_nw,
                           maxdd_ls_net     = st_n$max_dd,
                           avg_turnover_ls  = mean(bt$wide$turnover_ls, na.rm = TRUE)
                         ), fill = TRUE)
    fwrite(bt$wide,
           file.path(OUT_DIR_REFIT,
                     paste0("predsort_timeseries_", pc, "_year_", yy,
                            "_", RUN_TS, ".csv")))
  }

  # 儲存當年 NA tracking
  na_track_year <- rbindlist(na_track)
  fwrite(na_track_year,
         file.path(OUT_DIR_REFIT, paste0("na_tracking_year_", yy, ".csv")))

  cat("✅ Year", yy, "done.\n\n")
}


# ============================================================
# 12) 儲存預測 Panel + FULL OOS 彙總 + 逐年 R²_OOS
# ============================================================

pred_dt <- rbindlist(pred_all, fill = TRUE)
if (nrow(pred_dt) == 0) stop("No predictions produced.")

# 儲存完整預測 panel
pred_panel_path <- file.path(OUT_DIR_REFIT,
                              paste0("pred_panel_ANNUALREFIT_ED_", RUN_TS, ".csv"))
fwrite(pred_dt, pred_panel_path)

# 儲存逐年績效彙總
pred_sum_path <- file.path(OUT_DIR_REFIT,
                            paste0("predsort_summary_BYYEAR_", RUN_TS, ".csv"))
setorder(pred_bt_sum, year_refit, -sharpe_ls_net)
fwrite(pred_bt_sum, pred_sum_path)

# 合併 OOS NA tracking
na_files <- list.files(OUT_DIR_REFIT, pattern = "^na_tracking_year_",
                        full.names = TRUE)
if (length(na_files) > 0) {
  na_all     <- rbindlist(lapply(na_files, fread))
  na_summary <- na_all[, .(mean_na_rate = mean(na_rate),
                             sd_na_rate   = sd(na_rate)), by = step]
  fwrite(na_all,
         file.path(OUT_DIR_REFIT,
                   paste0("na_tracking_all_years_", RUN_TS, ".csv")))
  fwrite(na_summary,
         file.path(OUT_DIR_REFIT,
                   paste0("na_tracking_summary_", RUN_TS, ".csv")))
  cat("\n【NA Tracking OOS 全樣本平均】\n"); print(na_summary)

  # 合併訓練期（2005–2015）+ OOS（2016–2025）→ 完整 2005–2025 趨勢
  train_byyear_path <- file.path(
    OUT_DIR_REFIT,
    paste0("na_tracking_TRAIN_expanding_byyear_", RUN_TS, ".csv"))
  if (file.exists(train_byyear_path)) {
    na_train_for_merge <- fread(train_byyear_path)
    na_oos_wide <- dcast(
      na_all[step %in% c("1_raw_drop", "2_locf", "3_cs_median"),
             .(year = year, step, na_rate)],
      year ~ step, value.var = "na_rate"
    )
    setnames(na_oos_wide,
             c("1_raw_drop", "2_locf", "3_cs_median"),
             c("step_1_raw", "step_2_locf", "step_3_cs_median"),
             skip_absent = TRUE)
    na_train_slim    <- na_train_for_merge[, .(year, step_1_raw,
                                                step_2_locf, step_3_cs_median)]
    na_full_trend    <- rbind(na_train_slim, na_oos_wide, fill = TRUE)
    setorder(na_full_trend, year)
    fwrite(na_full_trend,
           file.path(OUT_DIR_REFIT,
                     paste0("na_tracking_FULL_2005_2025_", RUN_TS, ".csv")))
    cat("\n【完整 2005–2025 NA rate 趨勢（論文第四章用）】\n")
    print(na_full_trend)
  }
}

# FULL OOS 彙總（所有 119 個月合併）
pred_cols_all <- grep("^pred_", names(pred_dt), value = TRUE)
full_sum <- data.table()
for (pc in pred_cols_all) {
  bt   <- run_pred_sort_bt(pred_dt, score_col = pc, n_bucket = N_BUCKET,
                            trading_cost = TRADING_COST)
  st_g <- perf_stats(bt$wide$ls_gross)
  st_n <- perf_stats(bt$wide$ls_net)
  full_sum <- rbind(full_sum,
                    data.table(
                      model          = pc,
                      n_months       = st_n$n,
                      ann_ret_gross  = st_g$ann_ret,
                      sharpe_gross   = st_g$sharpe,
                      ann_ret_net    = st_n$ann_ret,
                      sharpe_net     = st_n$sharpe,
                      tstat_net      = st_n$tstat,
                      tstat_nw_net   = st_n$tstat_nw,
                      maxdd_net      = st_n$max_dd,
                      avg_turnover_ls= mean(bt$wide$turnover_ls, na.rm = TRUE)
                    ), fill = TRUE)
  fwrite(bt$wide,
         file.path(OUT_DIR_REFIT,
                   paste0("predsort_FULL_OOS_", pc, "_", RUN_TS, ".csv")))
}
setorder(full_sum, -sharpe_net)
full_sum_path <- file.path(OUT_DIR_REFIT,
                            paste0("predsort_FULL_OOS_summary_", RUN_TS, ".csv"))
fwrite(full_sum, full_sum_path)

# 逐年 R²_OOS（對應論文表 10）
cat("\n[補充] 逐年 R²_OOS\n")
r2_byyear <- data.table()
for (yy in years) {
  dt_yr <- pred_dt[year_refit == yy]
  if (nrow(dt_yr) == 0) next
  for (pc in pred_cols_all) {
    dt_sub <- dt_yr[is.finite(y) & is.finite(get(pc))]
    if (nrow(dt_sub) < 10) next
    ss_res <- sum((dt_sub$y - dt_sub[[pc]])^2)
    ss_tot <- sum((dt_sub$y)^2)   # R̄ = 0 基準
    r2_byyear <- rbind(r2_byyear,
                       data.table(year = yy, model = pc,
                                  r2_oos = round((1 - ss_res / ss_tot) * 100, 3)))
  }
}
fwrite(r2_byyear,
       file.path(OUT_DIR_REFIT, paste0("r2_oos_byyear_", RUN_TS, ".csv")))
cat("逐年 R²_OOS 已儲存。\n")


# ============================================================
# 13) 符號修正穩健性比較（對應論文 5.5 節）
# ============================================================

cat("\n[8D] 符號修正穩健性：ls_mean vs ls_tstat\n")

# 各年度符號修正一致性比較
sign_files_main <- list.files(OUT_DIR_REFIT,
                               pattern = "factor_signs_train_year_[0-9]{4}\\.csv",
                               full.names = TRUE)
sign_files_rob  <- list.files(OUT_DIR_REFIT,
                               pattern = "factor_signs_train_year_[0-9]{4}_ROBUST_tstat\\.csv",
                               full.names = TRUE)

if (length(sign_files_main) > 0 && length(sign_files_rob) > 0) {
  sign_compare <- rbindlist(lapply(seq_along(sign_files_main), function(i) {
    d1 <- fread(sign_files_main[i]); d2 <- fread(sign_files_rob[i])
    yr <- as.integer(gsub(".*year_(\\d{4})\\.csv", "\\1",
                          basename(sign_files_main[i])))
    data.table(year      = yr,
               n_factors = nrow(d1),
               n_agree   = sum(d1$sign == d2$sign),
               all_agree = all(d1$sign == d2$sign))
  }))
  setorder(sign_compare, year)
  fwrite(sign_compare,
         file.path(OUT_DIR_REFIT,
                   paste0("sign_method_compare_byyear_", RUN_TS, ".csv")))
  cat("符號修正一致性比較：\n"); print(sign_compare)
  cat(sprintf("全部年度完全一致：%s\n",
              ifelse(all(sign_compare$all_agree), "是", "否")))
}

# FULL OOS 符號穩健性績效比較
rob_models <- intersect(c("pred_sofar_comp_main", "pred_sofar_comp_rob"),
                         pred_cols_all)
rob_sum <- data.table()
for (pc in rob_models) {
  bt   <- run_pred_sort_bt(pred_dt, score_col = pc, n_bucket = N_BUCKET,
                            trading_cost = TRADING_COST)
  st_n <- perf_stats(bt$wide$ls_net)
  rob_sum <- rbind(rob_sum,
                   data.table(
                     model          = pc,
                     n_months       = st_n$n,
                     ann_ret_net    = st_n$ann_ret,
                     sharpe_net     = st_n$sharpe,
                     tstat_net      = st_n$tstat,
                     tstat_nw_net   = st_n$tstat_nw,
                     maxdd_net      = st_n$max_dd,
                     avg_turnover_ls= mean(bt$wide$turnover_ls, na.rm = TRUE)
                   ), fill = TRUE)
}
fwrite(rob_sum,
       file.path(OUT_DIR_REFIT,
                 paste0("sign_robustness_summary_FULL_OOS_", RUN_TS, ".csv")))
cat("符號穩健性結果：\n"); print(rob_sum)


# ============================================================
# 14) XGBoost-Hybrid 特徵重要性（對應論文 5.4 節）
#     以最後一年訓練期（2005–2024）資料重跑
# ============================================================

cat("\n[特徵重要性] XGBoost-Hybrid（2005–2024 全訓練期）\n")
set.seed(GLOBAL_SEED)

dtr_imp  <- xgboost::xgb.DMatrix(data = Xh_tr_final, label = y_tr_final)
params_imp <- list(
  objective        = "reg:squarederror",
  eval_metric      = "rmse",
  eta              = 0.05,
  max_depth        = 4,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  seed             = GLOBAL_SEED
)
bst_imp       <- xgboost::xgb.train(params = params_imp, data = dtr_imp,
                                     nrounds = 300, verbose = 0)
importance_imp <- xgboost::xgb.importance(feature_names = hybrid_cols_final,
                                           model = bst_imp)

fwrite(importance_imp,
       file.path(OUT_DIR_REFIT,
                 paste0("xgb_hybrid_importance_", RUN_TS, ".csv")))
cat("XGBoost 特徵重要性已儲存。\n"); print(importance_imp)


# ============================================================
# 15) 跨越檢定（Spanning Test，對應論文 5.2.3 節）
# ============================================================
gc()  # 強制垃圾回收，釋放記憶體
cat("\n[跨越檢定] XGBoost-Hybrid ~ XGBoost-Market + XGBoost-SOFAR\n")

bt_H <- run_pred_sort_bt(pred_dt, score_col = "pred_xgb_H",
                          n_bucket = N_BUCKET, trading_cost = TRADING_COST)
bt_M <- run_pred_sort_bt(pred_dt, score_col = "pred_xgb_M",
                          n_bucket = N_BUCKET, trading_cost = TRADING_COST)
bt_S <- run_pred_sort_bt(pred_dt, score_col = "pred_xgb_S",
                          n_bucket = N_BUCKET, trading_cost = TRADING_COST)

dt_span <- merge(bt_H$wide[, .(ym, r_H = ls_net)],
                  bt_M$wide[, .(ym, r_M = ls_net)], by = "ym")
dt_span <- merge(dt_span,
                  bt_S$wide[, .(ym, r_S = ls_net)], by = "ym")
dt_span <- dt_span[is.finite(r_H) & is.finite(r_M) & is.finite(r_S)]

fit_span    <- lm(r_H ~ r_M + r_S, data = dt_span)
span_result <- lmtest::coeftest(
  fit_span,
  vcov = sandwich::NeweyWest(fit_span, lag = NW_LAG, prewhite = FALSE))

# 直接從 span_result 矩陣取值
span_dt <- data.table(
  term      = rownames(span_result),
  estimate  = span_result[, 1],
  std_error = span_result[, 2],
  t_stat    = span_result[, 3],
  p_value   = span_result[, 4]
)
colnames(span_dt) <- c("term", "estimate", "std_error", "t_stat", "p_value")
fwrite(span_dt,
       file.path(OUT_DIR_REFIT,
                 paste0("spanning_test_result_", RUN_TS, ".csv")))
cat("跨越檢定結果：\n"); print(span_result)


# ============================================================
# 16) PCA 對照實驗（對應論文附錄 B）
# ============================================================

cat("\n========================================\n")
cat("[PCA 對照] XGBoost-PCA-S / XGBoost-PCA-H（ED 動態因子數）\n")
cat("========================================\n")

pred_all_pca  <- list()
pred_bt_sum_pca <- data.table()

for (yy in years) {
  set.seed(GLOBAL_SEED)
  cat("PCA-XGBoost YEAR:", yy, "\n")

  train_end  <- as.Date(paste0(yy - 1, "-12-01"))
  test_start <- as.Date(paste0(yy,     "-01-01"))
  test_end   <- as.Date(paste0(yy,     "-12-01"))

  dsub <- df[universe == TRUE & ym >= TRAIN_START & ym <= test_end]
  dtr0 <- dsub[ym <= train_end]
  dte0 <- dsub[ym >= test_start & ym <= test_end]
  if (nrow(dtr0) == 0 || nrow(dte0) == 0) { cat("skip\n"); next }

  x_vars_pca <- intersect(x_vars_all, names(dtr0))

  # --- 前處理（同主流程）---
  dt_u <- rbind(dtr0[, c("stock_id", "ym", x_vars_pca), with = FALSE],
                dte0[, c("stock_id", "ym", x_vars_pca), with = FALSE], fill = TRUE)
  setorder(dt_u, stock_id, ym)

  # Hard Drop
  dt_tr_raw   <- dt_u[ym <= train_end]
  na_rate_raw <- sapply(x_vars_pca, function(v) mean(!is.finite(dt_tr_raw[[v]])))
  drop_raw_p  <- names(na_rate_raw)[na_rate_raw > RAW_NA_CUT]
  if (length(drop_raw_p) > 0) {
    x_vars_pca <- setdiff(x_vars_pca, drop_raw_p)
    dt_u <- dt_u[, c("stock_id", "ym", x_vars_pca), with = FALSE]
  }
  if (length(x_vars_pca) < 5) { cat("skip: too few vars\n"); next }

  # LOCF
  for (v in x_vars_pca)
    dt_u[, (v) := nafill(get(v), type = "locf"), by = stock_id]

  # CS Median
  for (v in x_vars_pca) {
    dt_u[, tmp_med := {
      x <- as.numeric(get(v)); ok <- is.finite(x)
      if (sum(ok) < MIN_CS) NA_real_ else median(x[ok], na.rm = TRUE)
    }, by = ym]
    dt_u[!is.finite(get(v)), (v) := tmp_med]
    dt_u[, tmp_med := NULL]
  }

  # Winsorize + Z-score
  for (v in x_vars_pca) dt_u[, (v) := winsorize(get(v)), by = ym]
  for (v in x_vars_pca) dt_u[, (v) := cs_z(get(v)),      by = ym]

  # Soft Drop
  dt_tr_post  <- dt_u[ym <= train_end]
  na_rate_post_p <- sapply(x_vars_pca,
                            function(v) mean(!is.finite(dt_tr_post[[v]])))
  drop_post_p <- names(na_rate_post_p)[na_rate_post_p > POST_NA_CUT]
  if (length(drop_post_p) > 0) {
    x_vars_pca <- setdiff(x_vars_pca, drop_post_p)
    dt_u <- dt_u[, c("stock_id", "ym", x_vars_pca), with = FALSE]
  }

  # Fill zero
  for (v in x_vars_pca) dt_u[!is.finite(get(v)), (v) := 0]

  dt_tr_p <- dt_u[ym <= train_end]
  dt_te_p <- dt_u[ym >= test_start & ym <= test_end]
  X_tr_p  <- as.matrix(dt_tr_p[, ..x_vars_pca]); X_tr_p[!is.finite(X_tr_p)] <- 0
  X_te_p  <- as.matrix(dt_te_p[, ..x_vars_pca]); X_te_p[!is.finite(X_te_p)] <- 0

  # --- ED 動態決定 PCA 因子數（與 SOFAR 版本一致）---
  sv_p    <- svd(X_tr_p, nu = 0, nv = 0)
  lambda_p <- sort((sv_p$d^2) / (nrow(X_tr_p) - 1), decreasing = TRUE)
  ed_p    <- ed_select_r(lambda_p, r_max = R_MAX_ED,
                          tail = ED_TAIL, gap_mult = GAP_MULTIPLIER)
  n_pc    <- max(1, ed_p$r_hat)
  cat("ED selected n_pc:", n_pc, "\n")

  # --- PCA 估計 ---
  pca_fit <- prcomp(X_tr_p, center = FALSE, scale. = FALSE)
  n_pc    <- min(n_pc, ncol(pca_fit$rotation))
  V_pca   <- pca_fit$rotation[, 1:n_pc, drop = FALSE]
  F_pca_tr <- X_tr_p %*% V_pca
  F_pca_te <- X_te_p %*% V_pca
  colnames(F_pca_tr) <- paste0("pca_f", 1:n_pc)
  colnames(F_pca_te) <- paste0("pca_f", 1:n_pc)
  cat("PCA Var explained:",
      round(100 * sum(pca_fit$sdev[1:n_pc]^2) / sum(pca_fit$sdev^2), 1), "%\n")

  # --- 符號校正（同主流程邏輯）---
  merge_cols_p <- intersect(
    c("stock_id", "ym", "ret_fwd1", "mktcap_eom", "pb_tej",
      "mom_12m", "mom_1m", "mom_6m", "ret_vol", "log_turnover"),
    names(dtr0)
  )
  tr_y_p <- dtr0[, ..merge_cols_p]; te_y_p <- dte0[, ..merge_cols_p]
  tr_panel_p <- cbind(dt_tr_p[, .(stock_id, ym)], as.data.table(F_pca_tr))
  te_panel_p <- cbind(dt_te_p[, .(stock_id, ym)], as.data.table(F_pca_te))
  tr_panel_p <- merge(tr_panel_p, tr_y_p, by = c("stock_id", "ym"), all.x = TRUE)
  te_panel_p <- merge(te_panel_p, te_y_p, by = c("stock_id", "ym"), all.x = TRUE)
  tr_panel_p[, ret_fwd1_w := winsor_cs_ret(ret_fwd1, p = RET_WINS_P), by = ym]
  te_panel_p[, ret_fwd1_w := winsor_cs_ret(ret_fwd1, p = RET_WINS_P), by = ym]
  RET_COL_P <- "ret_fwd1_w"

  pca_raw_cols <- paste0("pca_f", 1:n_pc)
  for (fc in pca_raw_cols) {
    bt_tmp <- copy(tr_panel_p[is.finite(get(RET_COL_P))])
    bt_tmp[, bucket := assign_bucket(get(fc), n_bucket = N_BUCKET), by = ym]
    gret_tmp <- bt_tmp[!is.na(bucket),
                        .(ret = mean(get(RET_COL_P), na.rm = TRUE)),
                        by = .(ym, bucket)]
    wide_tmp <- dcast(gret_tmp, ym ~ bucket, value.var = "ret")
    setorder(wide_tmp, ym)
    ls_tmp  <- .safe_pick_col(wide_tmp, as.character(N_BUCKET)) -
      .safe_pick_col(wide_tmp, "1")
    ls_mean_p <- mean(ls_tmp[is.finite(ls_tmp)], na.rm = TRUE)
    sgn_p   <- ifelse(is.finite(ls_mean_p) && ls_mean_p < 0, -1L, 1L)
    scc_p   <- paste0(fc, "_sc")
    tr_panel_p[, (scc_p) := as.numeric(get(fc)) * sgn_p]
    te_panel_p[, (scc_p) := as.numeric(get(fc)) * sgn_p]
  }
  pca_sc_cols <- paste0(pca_raw_cols, "_sc")

  # --- 市場面特徵（同主流程）---
  for (v in market_wanted) {
    if (v %in% names(tr_panel_p)) tr_panel_p[, (v) := as.numeric(get(v))]
    if (v %in% names(te_panel_p)) te_panel_p[, (v) := as.numeric(get(v))]
  }
  for (v in market_wanted) {
    tr_panel_p[, (v) := winsor_cs2(get(v), p = WINS_P), by = ym]
    te_panel_p[, (v) := winsor_cs2(get(v), p = WINS_P), by = ym]
    tr_panel_p[, paste0("z_", v) := cs_z_score(get(v)), by = ym]
    te_panel_p[, paste0("z_", v) := cs_z_score(get(v)), by = ym]
  }
  market_z_p     <- intersect(paste0("z_", market_wanted), names(tr_panel_p))
  pca_hybrid_cols <- unique(c(pca_sc_cols, market_z_p))

  # --- XGBoost 訓練與預測 ---
  tr_ml_p <- tr_panel_p[is.finite(get(RET_COL_P))]
  te_ml_p <- te_panel_p[is.finite(get(RET_COL_P))]
  if (nrow(tr_ml_p) == 0 || nrow(te_ml_p) == 0) { cat("skip\n"); next }
  y_tr_p <- as.numeric(tr_ml_p[[RET_COL_P]])
  y_te_p <- as.numeric(te_ml_p[[RET_COL_P]])

  # PCA-only
  Xp_tr <- make_X(tr_ml_p, pca_sc_cols); Xp_te <- make_X(te_ml_p, pca_sc_cols)
  set.seed(GLOBAL_SEED)
  pred_xgb_pca_s <- .xgb_fit_predict(Xp_tr, y_tr_p, Xp_te)

  # PCA-Hybrid
  Xph_tr <- make_X(tr_ml_p, pca_hybrid_cols)
  Xph_te <- make_X(te_ml_p, pca_hybrid_cols)
  set.seed(GLOBAL_SEED)
  pred_xgb_pca_h <- .xgb_fit_predict(Xph_tr, y_tr_p, Xph_te)

  out_pca <- te_ml_p[, .(stock_id, ym)]
  out_pca[, y := y_te_p]; out_pca[, year_refit := yy]
  out_pca[, pred_xgb_PCA_S := pred_xgb_pca_s]
  out_pca[, pred_xgb_PCA_H := pred_xgb_pca_h]
  pred_all_pca[[as.character(yy)]] <- out_pca

  # 逐年回測
  for (pc_p in c("pred_xgb_PCA_S", "pred_xgb_PCA_H")) {
    bt_p   <- run_pred_sort_bt(out_pca, score_col = pc_p,
                                n_bucket = N_BUCKET, trading_cost = TRADING_COST)
    st_g_p <- perf_stats(bt_p$wide$ls_gross)
    st_n_p <- perf_stats(bt_p$wide$ls_net)
    pred_bt_sum_pca <- rbind(pred_bt_sum_pca,
                              data.table(
                                year_refit       = yy, model = pc_p,
                                ann_ret_ls_gross = st_g_p$ann_ret,
                                sharpe_ls_gross  = st_g_p$sharpe,
                                ann_ret_ls_net   = st_n_p$ann_ret,
                                sharpe_ls_net    = st_n_p$sharpe,
                                tstat_nw_ls_net  = st_n_p$tstat_nw,
                                maxdd_ls_net     = st_n_p$max_dd
                              ), fill = TRUE)
  }
  cat("✅ PCA-XGBoost Year", yy, "done.\n\n")
}

# PCA FULL OOS 彙總
pred_dt_pca <- rbindlist(pred_all_pca, fill = TRUE)
full_sum_pca <- data.table()
for (pc_p in c("pred_xgb_PCA_S", "pred_xgb_PCA_H")) {
  bt_p   <- run_pred_sort_bt(pred_dt_pca, score_col = pc_p,
                              n_bucket = N_BUCKET, trading_cost = TRADING_COST)
  st_n_p <- perf_stats(bt_p$wide$ls_net)
  full_sum_pca <- rbind(full_sum_pca,
                         data.table(
                           model        = pc_p,
                           n_months     = st_n_p$n,
                           ann_ret_net  = st_n_p$ann_ret,
                           sharpe_net   = st_n_p$sharpe,
                           tstat_nw_net = st_n_p$tstat_nw,
                           maxdd_net    = st_n_p$max_dd
                         ), fill = TRUE)
  fwrite(bt_p$wide,
         file.path(OUT_DIR_PCA,
                   paste0("predsort_FULL_OOS_", pc_p, "_", RUN_TS, ".csv")))
}
cat("\n====== PCA-XGBoost 全期績效 ======\n"); print(full_sum_pca)
fwrite(full_sum_pca,
       file.path(OUT_DIR_PCA,
                 paste0("pca_xgb_full_summary_", RUN_TS, ".csv")))
fwrite(pred_bt_sum_pca,
       file.path(OUT_DIR_PCA,
                 paste0("pca_xgb_byyear_summary_", RUN_TS, ".csv")))
fwrite(pred_dt_pca,
       file.path(OUT_DIR_PCA,
                 paste0("pred_panel_PCA_", RUN_TS, ".csv")))


# ============================================================
# 完成報告
# ============================================================

cat("\n========================================\n")
cat("✅ pipeline_main.R 全部完成\n")
cat("Run 資料夾  :", OUT_DIR, "\n")
cat("主流程輸出  :", OUT_DIR_REFIT, "\n")
cat("PCA 對照輸出:", OUT_DIR_PCA, "\n")
cat("Log 檔案   :", logfile, "\n")
cat("========================================\n")

try(sink(), silent = TRUE)
