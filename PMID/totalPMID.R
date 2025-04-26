# this code shows the total number of unique paper is 180
# master_unique$pmid_number will give out all the PMID number 
library(readxl)
library(dplyr)
library(janitor)
library(readr)

paths <- c(
  "Mediation-DataExtraction-Adam-Primary.xlsx",
  "Mediation-DataExtraction-Adam-Secondary.xlsx",
  "Mediation-DataExtraction-Gunho-Primary.xlsx",
  "Mediation-DataExtraction-Gunho-Secondary.xlsx",
  "Mediation-DataExtraction-Ian-Primary.xlsx",
  "Mediation-DataExtraction-Ian-Secondary.xlsx",
  "Mediation-DataExtraction-Jeannie-Primary_JMSL08032018.xlsx",
  "Copy of Mediation-DataExtraction-Jeannie-Secondary (1).xlsx",
  "Mediation-DataExtraction-Kara-Primary.xlsx",
  "Mediation-DataExtraction-Kara-Secondary.xlsx",
  "Mediation-DataExtraction-Kelly-Primary-completed.xlsx",
  "Mediation-DataExtraction-Kelly-Secondary-Completed.xlsx",
  "Mediation-DataExtraction-Liz-Primary.xlsx",
  "Mediation-DataExtraction-Liz-Secondary.xlsx",
  "Copy of Mediation-DataExtraction-Trang-Primary.xlsx",
  "Copy of Mediation-DataExtraction-Trang-Secondary.xlsx"
)

read_form <- function(path) {
  reviewer <- ifelse(grepl("-Primary", path, ignore.case = TRUE), "A", "B")
  df <- read_excel(path) |> clean_names()
  
  # If sample_size exists and is character, coerce it to numeric
  if ("sample_size" %in% names(df) && is.character(df$sample_size)) {
    df$sample_size <- readr::parse_number(df$sample_size)
  }
  
  df |> mutate(reviewer = reviewer)
}

# Use the read_form function when reading each file
dfs <- lapply(paths, read_form)

# Now bind all data frames together
master_df <- bind_rows(dfs, .id = "source")
master_df <- type_convert(master_df)

# Print all column names to find the correct one
print(colnames(master_df))

pmid_cols <- grep("pmid|pid", colnames(master_df), ignore.case = TRUE, value = TRUE)
print("Possible PMID columns:")
print(pmid_cols)
pmid_col <- pmid_cols[1]

master_unique <- master_df %>% 
  filter(!is.na(.data[[pmid_col]])) %>%
  distinct(.data[[pmid_col]], .keep_all = TRUE)

cat("Total extraction rows :", nrow(master_df), "\n")
cat("Unique PMID papers    :", nrow(master_unique), "\n")

head(master_df)
head(master_unique)


#library(writexl)

#write_xlsx(master_unique, "unique_pmid_papers.xlsx")
#cat("Excel file 'unique_pmid_papers.xlsx' has been created successfully.\n")