library(data.table)
library(sandwich)
library(lmtest)

dt_0050 <- fread("C:/我的/政大/碩二下/論文輸出資料與模型/0050.csv", 
                 encoding = "UTF-8")

# 處理日期
dt_0050[, date := as.Date(year, format = "%Y/%m/%d")]
dt_0050[, ym := format(date, "%Y-%m")]

# 月初第一個交易日
bom <- dt_0050[, .(close_bom = first(close)), by = ym]

# 月末最後一個交易日
eom <- dt_0050[, .(close_eom = last(close)), by = ym]

# 合併計算月報酬率
dt_ret <- merge(bom, eom, by = "ym")
dt_ret[, ret := close_eom / close_bom - 1]

# 篩選樣本外期間
dt_ret <- dt_ret[ym >= "2016-01" & ym <= "2025-11"]

r <- dt_ret$ret
mu <- mean(r)
# 年化報酬
ann_ret <- prod(1 + r)^(12/length(r)) - 1

# 年化波動度
ann_vol <- sd(r) * sqrt(12)

# 夏普比率
sharpe <- (mu * 12) / ann_vol


# Newey-West t統計量（lag=6，跟主pipeline一致）

df_nw <- data.frame(y = r)
fit_nw <- lm(y ~ 1, data = df_nw)
nw_se <- sqrt(as.numeric(
  sandwich::NeweyWest(fit_nw, lag = 6, prewhite = FALSE)[1, 1]))
tstat_nw <- mu / nw_se

# 最大回撤（跟論文公式14一致）
cum_ret <- cumprod(1 + r)
running_max <- cummax(cum_ret)
dd <- (cum_ret - running_max) / running_max
max_dd <- min(dd, na.rm = TRUE)

cat("0050年化報酬：", round(100*ann_ret, 3), "%\n")
cat("0050夏普比率：", round(sharpe, 3), "\n")
cat("0050 NW t統計量：", round(tstat_nw, 3), "\n")
cat("0050最大回撤：", round(100*max_dd, 3), "%\n")
cat("月數：", length(r), "\n")