## CausalJudge: a project on LLM Evaluation of Causal Claims and Methodological Assumptions


- Dropbox link to the data: https://www.dropbox.com/scl/fo/cnybhopuhocm339vp29dq/AM5fGE_1yYiNjUTIsci9GnY?rlkey=qqpdq45opol5dgyiyt69ue36s&st=qii4ntuj&dl=0. Right now this only has Cohrane Review Data with an example zip file pushed to the repo. Basically, after unarchiving, you will get the underlying metadata and analysis that the folks have performed for these meta-analyses.

### TODO:
I will use this section to track what things we wanted to do to move this project forward (in rough order of importance?):
#### For the causal mediation analysis 
- [X] [Downloaded pdf](https://www.dropbox.com/scl/fo/8s4hvd01ar1m2zexbdajl/AJVOrh7nSYOFYwo5N9CNgxI?rlkey=r3stigqixv7vtezi8xpyyul3a&dl=0) ~~Download all pdfs used in the review (is this already done? If not we can easily do it by either searching it on PubMed and/or grab it from the publisher).~~
- [X] ~~Extract all texts/table from the paper.~~ Saved in `PMID_all_text.jsonl`.
- [X] ~~Adapt Cathy's code pipeline to generate structured output like `json` so it's easier to parse and analyze.~~ Preliminary code at `cathy-code-causal-mediation.ipynb`.
- [X] ~~Run the prompts to all the review data~~. Preliminary code at `cathy-code-causal-mediation.ipynb`.
- [X] ~~Pick a few papers for human review.~~ From meeting with Cathy, the automated extraction process yields good end-to-end results compared to manual copying and pasting. We should make sure to mention this in the write-up.
- [X] The binary decisions for all 180 articles—along with the gold standard and GPT’s responses under different prompts—are compiled in `180FinalResult_Jun17.xlsx` within the PMID folder. Supporting subfiles, such as `extracted_full_causalcode.csv`, `extracted_full_explain.csv,`, ``extracted_full_causaluncertain.csv`, and `extracted_full_causaluncertain2.csv`, are also included and together form the basis of the final result.  
- [X] Four Python files—`causalcode.py`, `causalexplain.py`, `causaluncertain.py`, and `causaluncertain2.py`—contain the code and corresponding prompts used in the analysis.
- [X] The scripts `1_dataframe_count_Jun16.R` through `6_ArticleswithReconcilation_Jun16.R`, are numbered R scripts used sequentially in the analysis workflow. 
- [ ] Write up the basic findings.
- [ ] Analyzing qualitative response using LLM and pick some for human review.
- [ ] Analyze accuracy stratified by journal/year of publication **it would be interesting to see if whether certain fields/topics are associated with lower accuracy; @cathy**

#### For the Cochrane databases:
- [ ] I think we should start tabulating some questions we can ask based on the downloaded data, and try to understand which ones are the most interesting (we can check in w/ Liz).
- [ ] Experiment with LLM-based assessment of risk of bias.
- [ ] Experiment with LLM-based extraction of the study characteristics.

#### More datasets:
TBD

### Functions:
1. `web-crawl-download-cochrane.py` is the function that runs a webcrawler using selenium that performs interactive crawling (it's annoying to scrape since we need to click on a bunch of buttons...). I don't think you need to interact with this function too much, and AI really wrote most of it lol. That said, good to have the reference to know how we might be able to include other information and datasets in the future.


### Some potentially helpful pointers:
* Extract text from pdf:
	* If there is a lot of math, I've had good experience with https://mathpix.com/pricing (you could try uploading one and see how it goes and then use API if needed). It costs $ but I am happy to reimburse you out of my research account. Email `BiostatTravel-AP@jh.edu` with your receipts and cc me.
	* There are some free tools such as `https://github.com/VikParuchuri/marker` and `https://github.com/opendatalab/PDF-Extract-Kit`. I think we need to automate the extraction -- and hopefully these are at least as good as copying and pasting...
* Unarchive all the zip files in a directory in Python
```
import zipfile
import os

# Set your target directory
zip_dir = '/path/to/your/zipfiles'

# Loop through all .zip files in the directory
for filename in os.listdir(zip_dir):
    if filename.endswith('.zip'):
        zip_path = os.path.join(zip_dir, filename)
        extract_dir = os.path.join(zip_dir, filename[:-4])  # Strip .zip to create a folder

        # Make output directory if it doesn't exist
        os.makedirs(extract_dir, exist_ok=True)

        # Extract zip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            print(f'Extracted {filename} to {extract_dir}')
```
