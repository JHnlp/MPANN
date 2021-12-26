# Multi-Probe Attention for Semantic Indexing

### About

This project is developed for the topic of COVID-19 semantic indexing.

### Directories & files

A. The directory of the raw CovSI Corpus: './data/covid19/'  
B. The directory of the MeSH vocabulary: './data/MeSH/'  
C. Source code modules: './biosemin/*'  
D. Main script file : './run_mpann.py'  
E. Initialization models: './model/'  
F. Output directory: './output/'

==============================================================

### Updates for BioASQ 2021 Task9a

1. The project has been updated to be compatable with the BIOASQ 2021 Task9a dataset.
2. The code has been modified to follow the typical structure of the pre-trained transformer encoder and is able to be
   initalized with BERT-based models.
3. **BioASQ Task9a dataset**:  
   (a) **Training set**: The original compressed file of the BioASQ Task9a training set consists of more than 15 million
   articles with the size of around 20G, and should be download from the [BioASQ](http://bioasq.org/) official website  
   (b) **Test set**: The BioASQ Task9a test set is comprised of more than 90k aritcles and split into 15 different
   batches. The Task9a test set should be download from the [BioASQ](http://bioasq.org/) official website  
   (c) Since the some MeSH annotations are missed in parts of the articles in the BioASQ Task9a test set and some PMIDs
   are invalid or deprecated, we filtered these abnormal articles and added necessary candidate labels for prediction.
   The filtered test dataset locates at './data/bioasq2021/test_filtered/'
5. The script 'shell_bioasq2021.sh' for prediction on the BioASQ Task9a test is added.
