# Monthly long-short net returns of XGBoost-Hybrid for 2020, 2021 and 2025 (thesis Appendix D.1).
# Run from the repository root, after pipeline_main.R has finished.
# Automatically uses the latest run_* folder under outputs/pipeline_runs.

library(data.table)

BASE_OUT_DIR <- "outputs/pipeline_runs"
run_dirs <- list.dirs(BASE_OUT_DIR, recursive = FALSE, full.names = TRUE)
run_dirs <- run_dirs[grepl("run_\\d{8}_\\d{6}$", basename(run_dirs))]
if (length(run_dirs) == 0) stop("No run_* folder found. Run pipeline_main.R first.")
LATEST_RUN <- tail(sort(run_dirs), 1)

bt_file <- list.files(
  file.path(LATEST_RUN, "annual_refit"),
  pattern = "predsort_FULL_OOS_pred_xgb_H.*\\.csv",
  full.names = TRUE
)[1]

bt <- fread(bt_file)
bt[, ym := as.Date(ym)]

# Keep the three years discussed in the thesis
bt_years <- bt[year(ym) %in% c(2020, 2021, 2025), .(ym, ls_net)]
bt_years[, ls_net_pct := round(100 * ls_net, 3)]
setorder(bt_years, ym)
print(bt_years)