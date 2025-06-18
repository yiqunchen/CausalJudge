# This is the third code file in Causal Judge Series
# master_df got from first code file, # compiled_df got from the second code file
# This is a subfile trying list out the 32 articles that left raw (not encoded into binary form)
library(dplyr)
library(tidyr)

master_pmids <- unique(master_df$`PMID#`)
compiled_pmids <- unique(compiled_df$`PMID#`)

unjudged_pmids <- setdiff(master_pmids, compiled_pmids)

cat("Number of accessible papers without golden standard judgement:", length(unjudged_pmids), "\n")
print(length(unjudged_pmids))

unjudged_df <- unique(master_df %>% filter(`PMID#` %in% unjudged_pmids))

vars_14 <- c(
  "Title",
  "Name of mediation method",
  "Exposure randomized?",
  "Causal mediation? (yes/no)",
  "If B+K, examine linear relationship b/w mediator and outcome?",
  "If Baron+Kenny, examine whether no interaction b/w tx and mediator on outcome?",
  "Covariates in exposure/mediator model",
  "Covariates in exposure/outcome model",
  "Covariates in mediator/outcome model",
  "Baseline value of mediator adjusted for?",
  "Baseline value of outcome adjusted for?",
  "Temporal ordering of exposure before mediator (yes/no)",
  "Temporal ordering of mediator before outcome (yes/no)",
  "Discussion of mediation assumptions (yes/no)",
  "Sensitivity analyes to assumptions (yes/no)",
  "Does model control for other post-exposure variables (yes/no)"
)

#Gives out raw for 32 articles
unjudged_14 <- master_df %>%
  filter(`PMID#` %in% unjudged_pmids) %>%
  select(`PMID#`, all_of(vars_14))

paired_df <- unjudged_14 %>%
  add_count(`PMID#`, name = "n_in_pmid") %>%
  filter(n_in_pmid > 1) %>%
  select(-n_in_pmid) %>%
  group_by(`PMID#`) %>%
  mutate(row_in_pmid = row_number()) %>% 
  ungroup() %>%                                   
  pivot_wider(
    id_cols   = `PMID#`,
    names_from  = row_in_pmid,
    values_from = -c(`PMID#`, row_in_pmid),
    names_glue  = "{.value}_{row_in_pmid}"
  )

write_csv(paired_df, "paired_32_df.csv")

paired_df$`PMID#`




