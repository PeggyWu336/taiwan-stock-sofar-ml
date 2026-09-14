library(data.table)

bt_file <- list.files(
  "C:/我的/政大/碩二下/論文輸出資料與模型/pipeline_runs/run_20260527_222704/annual_refit",
  pattern = "predsort_FULL_OOS_pred_xgb_H.*\\.csv",
  full.names = TRUE
)[1]

bt <- fread(bt_file)
bt[, ym := as.Date(ym)]

# 篩選三年
bt_years <- bt[year(ym) %in% c(2020, 2021, 2025),
               .(ym, ls_net)]
bt_years[, ls_net_pct := round(100 * ls_net, 3)]
setorder(bt_years, ym)
print(bt_years)