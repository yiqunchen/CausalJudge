# Second code file in CausalJudge Series
# Examines we have golden standard judgement on how many papers
# Answer is 148

library(readr)
library(dplyr)

files <- c(
  "Adam_Ian_mediationreview_FINAL.csv",
  "Adam_Jeannie_mediationreview.csv",
  "Adam_Kara_mediationreview.csv",
  "Adam_Kelly_mediationreview.csv",
  "Jeannie_Ian_plusgunhoredo_mediationreview.csv",
  "Jeannie_Kara_mediationreview.csv",
  "Kara_Ian_mediationreview_FINAL.csv",
  "Kelly_Ian_mediationreview_FINAL.csv",
  "Trang_Ian_mediationreview_FINAL.csv"
)

compiled_df <- files %>%
  lapply(read_csv) %>%
  bind_rows()

cat("Total number of rows across all files:", nrow(compiled_df), "\n")

pmid_col <- grep("pmid", names(compiled_df), ignore.case = TRUE, value = TRUE)[1]
if (!is.null(pmid_col)) {
  unique_pmids <- compiled_df %>% filter(!is.na(.data[[pmid_col]])) %>% distinct(.data[[pmid_col]])
  cat("Unique PMIDs:", nrow(unique_pmids), "\n")
}

compiled_df[compiled_df$`PMID#` == 22998852, ]
