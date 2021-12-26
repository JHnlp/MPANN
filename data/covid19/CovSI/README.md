## CovSI Corpus

This description is mainly about the [COVID-19 Semantic Indexing (CovSI) Corpus](https://drive.google.com/file/d/11Z0v_q2m0Tsvc-hsQJZM6hckw_u87bEi/view?usp=sharing). The corpus is designed for the
semantic indexing research in COVID-19 domain.

## Corpus Construction

The corpus is constructed mainly based on three large resources of [CORD-19](https://allenai.org/data/cord-19)
, [MEDLINE](https://pubmed.ncbi.nlm.nih.gov/), and [PMC](https://www.ncbi.nlm.nih.gov/pmc/).

As the [COVID-19 Open Research Dataset (CORD-19)](https://allenai.org/data/cord-19) provides the largest dataset of
COVID-19 relevant articles, it is natural to be considered the fundamental resource For [CovSI](https://drive.google.com/file/d/11Z0v_q2m0Tsvc-hsQJZM6hckw_u87bEi/view?usp=sharing) due to its large coverage
and public accessibility. However, although CORD-19 carries lots of fundamental ingredients, it doesn’t provide any
relevant annotations for semantic indexing. Therefore, databases of [MEDLINE](https://pubmed.ncbi.nlm.nih.gov/)
and [PMC](https://www.ncbi.nlm.nih.gov/pmc/) are considered as the preferred supplementation.

In this regard, all kinds of attribute fields are first extracted from these resources, then the redundant information
is filtered and the reserved metadata is mapped. All extracted metadata is finally merged to construct the corpus of
[CovSI](https://drive.google.com/file/d/11Z0v_q2m0Tsvc-hsQJZM6hckw_u87bEi/view?usp=sharing) for COVID-19 semanticn indexing. Note that during the corpus construction, PMIDs/PMCIDs are treated as the unique
identifiers for articles, and the articles without valid PMIDs or PMCIDs would be discarded.

The [CovSI](https://drive.google.com/file/d/11Z0v_q2m0Tsvc-hsQJZM6hckw_u87bEi/view?usp=sharing) corpus is stored in the JSON format, consisting of ~87k COVID-19 related articles.

### The Attribute Statistics of CovSI

|Attribute| Count|
|:---:| :---: |
|PMID|87,207|
|PMCID|46,487|
|Title|87,192|
|Abstract|87,162|
|Body Text|45,968|
|MeSH Terms|1,161,962|
|MeSH Identifiers|1,161,962|
|Journal Name|87,207|
|Year|87,207|
|Authors|87,128|
|Affiliations|83,749|
|Keywords|35,928|
|Chemicals|43,711|
|DOI|77,776|
|URL|87,207|

### The Basic Statistics of the different CovSI subsets

|Dataset| #Articles| #Term Types| #Total Terms| #Average Terms| 
|:---:| :---: | :---: | :---: | :---: |
| <div style="width: 100pt"> Training set | <div style="width: 80pt">71,207 | <div style="width: 80pt">17,758 | <div style="width: 80pt"> 945,462 |<div style="width: 100pt"> 13.28|
|  Development set     | 8,000 | 9,035 | 106,088      | 13.26|
|  Test set | 8,000 | 8,991 | 110,412      | 13.80|

## JSON Data Structure

An example in JSON format of CovSI:

```json
{
  "pmid": "33107714",
  "pmcid": "",
  "title": "Management of COVID-19 in an Outpatient Dialysis Program.",
  "abstractText": "In March 2020, the COVID-19 pandemic became an increasingly urgent issue of public health ...",
  "meshMajor": [
    "Ambulatory Care",
    "COVID-19",
    "Coronavirus Infections",
    "Humans",
    "Infection Control",
    "Pandemics",
    "Pneumonia, Viral",
    "Renal Dialysis",
    "Retrospective Studies",
    "United States"
  ],
  "labels": [
    "D000553",
    "D000086382",
    "D018352",
    "D006801",
    "D017053",
    "D058873",
    "D011024",
    "D006435",
    "D012189",
    "D014481"
  ],
  "authors": "E Noce;M Zorzanello;D Patel;R Kodali",
  "journal": "Nephrology nursing journal : journal of the American Nephrology Nurses' Association",
  "medline_ta": "Nephrol Nurs J",
  "affiliations": "Nurse Practitioner, Yale-New Haven Hospital, Department of Internal Medicine, Section of Nephrology, New Haven, Tn.;Nurse Practitioner, Yale University, Department of Medicine, Section of Nephrology, New Haven, CT.;Nephrology Fellow, Yale University, Department of Medicine, Section of Nephrology, New Haven, CT.;Instructor of Medicine, Yale University, Department of Medicine, Section of Nephrology, New Haven, CT.",
  "keywords": "COVID-19; hemodialysis; infection control; pandemic",
  "year": "2020",
  "chemical_list": "",
  "doi": "",
  "url": "https://www.ncbi.nlm.nih.gov/pubmed/33107714/",
  "bodyText": ""
}
```









