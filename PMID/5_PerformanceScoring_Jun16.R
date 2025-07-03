# Fifth code file in CausalJudge Series
# Help to create table 1 and 2 (originally under a total of 148 articles)

library(readxl)
library(tidyverse)

dat_raw <- read_xlsx("148FinalResult_Jun15.xlsx")

method_col   <- "Type"
id_col       <- "PMID"

crit_cols <- c(
  "Randomized Exposure",
  "Control for Baseline Mediator",
  "Control for Baseline Outcome",
  "Sensitivity Analysis to Assumption",
  "Control for Other Post-Exposure Variables",
  "Examined Mediator-Outcome Linearity",
  "Examined Exposure-Mediator Interaction",
  "Covariates in Exposure-Mediator Model",
  "Covariates in Exposure-Outcome Model",
  "Covariates in Mediator-Outcome Model",
  "Temporal Ordering Exposure Before Mediator",
  "Temporal Ordering Mediator Before Outcome",
  "Discussed Mediator Assumptions",
  "Causal Mediation"
)

clean_cell <- function(x) {
  str_trim(x) %>%
    str_replace("^([01]).*$", "\\1")
}

long <- dat_raw %>%
  select(all_of(c(id_col, method_col, crit_cols))) %>%
  pivot_longer(cols = all_of(crit_cols),
               names_to  = "criterion",
               values_to = "value_raw") %>%
  mutate(value = clean_cell(value_raw))

gold <- long %>% filter(.data[[method_col]] == "original") %>%
  select(all_of(c(id_col, "criterion", "value"))) %>%
  rename(gold = value)

scored <- long %>%
  filter(.data[[method_col]] %in%
           c("basic", "explain", "uncertain", "uncertain2")) %>%
  select(-value_raw) %>%
  left_join(gold, by = c(id_col, "criterion")) %>%
  mutate(correct = (value == gold))

table1 <- scored %>%
  group_by(.data[[method_col]]) %>%
  summarise(correct = sum(correct),
            total   = n(),
            accuracy = sprintf("%.1f %%", 100 * correct / total),
            .groups = "drop") %>%
  arrange(match(.data[[method_col]],
                c("basic", "explain", "uncertain", "uncertain2")))

print(table1, n = Inf)


table2 <- scored %>%
  group_by(criterion, .data[[method_col]]) %>%
  summarise(correct = sum(correct),
            total   = n(),
            acc = 100 * correct / total,
            .groups = "drop") %>%
  mutate(acc_pct = sprintf("%.1f %%", acc)) %>%
  pivot_wider(names_from  = .data[[method_col]],
              values_from = acc_pct) %>%
  arrange(match(criterion, crit_cols))

print(table2, n = Inf)

