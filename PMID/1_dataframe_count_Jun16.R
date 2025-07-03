# First code file in CausalJudge Series
# Examines how many paper are still accessible to us and can be automatically extracted. 
# Answer would be 180

library(readxl)
library(tidyverse)

A1 <- read_excel("Mediation-DataExtraction-Adam-Primary.xlsx")
A2 <- read_excel("Mediation-DataExtraction-Adam-Secondary.xlsx")
G1 <- read_excel("Mediation-DataExtraction-Gunho-Primary.xlsx")
G2 <- read_excel("Mediation-DataExtraction-Gunho-Secondary.xlsx")
I1 <- read_excel("Mediation-DataExtraction-Ian-Primary.xlsx")
I2 <- read_excel("Mediation-DataExtraction-Ian-Secondary.xlsx")
J1 <- read_excel("Mediation-DataExtraction-Jeannie-Primary_JMSL08032018.xlsx")
J2 <- read_excel("Copy of Mediation-DataExtraction-Jeannie-Secondary (1).xlsx")
K1 <- read_excel("Mediation-DataExtraction-Kara-Primary.xlsx")
K2 <- read_excel("Mediation-DataExtraction-Kara-Secondary.xlsx")
KB1 <- read_excel("Mediation-DataExtraction-Kelly-Primary-completed.xlsx")
KB2 <- read_excel("Mediation-DataExtraction-Kelly-Secondary-Completed.xlsx")
L1 <- read_excel("Mediation-DataExtraction-Liz-Primary.xlsx")
L2 <- read_excel("Mediation-DataExtraction-Liz-Secondary.xlsx")
T1 <- read_excel("Copy of Mediation-DataExtraction-Trang-Primary.xlsx")
T2 <- read_excel("Copy of Mediation-DataExtraction-Trang-Secondary.xlsx")

A1$Reviewer <- "A"
A2$Reviewer <- "B"
G1$Reviewer <- "A"
G2$Reviewer <- "B"
I1$Reviewer <- "A"
I2$Reviewer <- "B"
J1$Reviewer <- "A"
J2$Reviewer <- "B"
K1$Reviewer <- "A"
K2$Reviewer <- "B"
KB1$Reviewer <- "A"
KB2$Reviewer <- "B"
L1$Reviewer <- "A"
L2$Reviewer <- "B"
T1$Reviewer <- "A"
T2$Reviewer <- "B"

G1$`Link to paper` <- NA
G1$`Reviewer 1__1` <- NA
G1$`Reviewer 2__1` <- NA
G2$`Link to paper` <- NA
G2$`Reviewer 1__1` <- NA
G2$`Reviewer 2__1` <- NA

colnames(K2)[25] <- "If causal, which estimand:   1) controlled direct and indirect, 2) natural direct and indirect effects, or 3) stochastic/randomized interventional direct and indirect effects"

L1$X__1 <- NULL
KB1$X__1 <- NULL
KB2$X__1 <- NULL

standardize_sample_size <- function(df) {
  if ("Sample size" %in% colnames(df)) {
    df$`Sample size` <- as.character(df$`Sample size`)
  }
  return(df)
}

all_dfs <- list(A1, A2, G1, G2, I1, I2, J1, J2, K1, K2, KB1, KB2, L1, L2, T1, T2)
all_dfs <- lapply(all_dfs, standardize_sample_size)

master_df <- bind_rows(all_dfs)
print(paste("Total rows in master dataframe:", nrow(master_df)))


library(writexl)

write_xlsx(master_df, path = "master.xlsx")


sum(sapply(all_dfs, nrow))
length(unique(master_df$`PMID#`))

#master_df[master_df$`PMID#` == 22998852, ]
