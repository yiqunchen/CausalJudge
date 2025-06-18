# Fourth code file in CausalJudge Series
# Small helper functions to help filter out and rerun some of the entries

library(tidyverse)

pmids_to_keep <- c(25012409, 24294875, 27124379, 26785063, 
                   23815233, 26524001, 29595292)

pmid_text_output <- read_csv("pmid_text_output.csv")

pmid_text_output_Jun14_RemediateFirst3 <- pmid_text_output %>%
  filter(`pmid` %in% pmids_to_keep)

write_csv(pmid_text_output_Jun14_Remediate, "pmid_text_output_Jun14_RemediateFirst3.csv")

pmid_text_output_Jun14_RemediateFirst3

#---------------------
library(tidyverse)

pmids_to_keep <- c(
  25012409, 24294875, 27124379, 26590510, 29510354, 25534593, 26785063,
  28768312, 25866294, 28872672, 26670947, 24485197, 26126419, 28956520,
  27554198, 24820537, 25111250, 27242616, 25640022, 27775415, 23453673,
  28477504, 25888339, 26395975, 27774075, 27252666, 23815233, 28193310,
  28647670, 29172592, 24012069, 28778553, 27219531, 27884931, 29202885,
  28871234, 26192040, 28611704, 26524001, 25039961, 27933010, 29294372,
  27267323, 28287798, 25071670, 27280309, 27378988, 27826274, 28729357,
  29477589, 25528759, 28229495, 25031222, 24239131, 23181545, 27799916,
  28843915, 29595292, 26098581, 24220644
)

pmid_text_output <- read_csv("pmid_text_output.csv")

pmid_text_output_Jun14_RemediateLast1 <- pmid_text_output %>%
  filter(`pmid` %in% pmids_to_keep)

write_csv(pmid_text_output_Jun14_RemediateLast1, "pmid_text_output_Jun14_RemediateLast1.csv")

pmid_text_output_Jun14_RemediateLast1


#--------------
library(tidyverse)

pmids_to_keep <- c(
  28843081, 26822489, 28880726, 27252674, 25998280, 28961425
)

pmid_text_output <- read_csv("pmid_text_output.csv")

pmid_text_output_Jun14_ExplainRemediate <- pmid_text_output %>%
  filter(`pmid` %in% pmids_to_keep)

write_csv(pmid_text_output_Jun14_ExplainRemediate, "pmid_text_output_Jun14_ExplainRemediate.csv")

pmid_text_output_Jun14_ExplainRemediate


#------------32 articles
library(tidyverse)

pmids_to_keep <- paired_df$`PMID#`

pmid_text_output <- read_csv("pmid_text_output.csv")

pmid_text_output_Jun16_32Articles <- pmid_text_output %>%
  filter(`pmid` %in% pmids_to_keep)

write_csv(pmid_text_output_Jun16_32Articles, "pmid_text_output_Jun16_32Articles.csv")

pmid_text_output_Jun16_32Articles
