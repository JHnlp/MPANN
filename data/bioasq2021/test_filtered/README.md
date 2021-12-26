## BioASQ 2021 Task9a datasets

The original BioASQ 2021 Task9a dataset could be downloaded from the official website of [BioASQ](http://bioasq.org/).

Since some articles in the original BioASQ Task9a test dataset do not have valid PMIDs or MeSH annotations, we filtered
these abnormal records to conduct the semantic indexing experiments.

a). The [filtered dataset](https://drive.google.com/file/d/1v5Vu5kBDY_3ajwWGhOa6LILN6w5Cbkh2/view?usp=sharing) of BioASQ
Task9a test dataset should be put in the directory of **'test_filtered'**.

b). All articles in
the [filtered dataset](https://drive.google.com/file/d/1v5Vu5kBDY_3ajwWGhOa6LILN6w5Cbkh2/view?usp=sharing) are stored in
JSON format line by line, and each line stands for a single article.

c). All articles in
the [filtered dataset](https://drive.google.com/file/d/1v5Vu5kBDY_3ajwWGhOa6LILN6w5Cbkh2/view?usp=sharing)  have been
added a new key of '**candidata_labels*' that provides the necessary information of candidate terms for semantic
indexing;

### Data Format Description

All reserved articles follow the JSON format below:

Example:

```json
{
  "pmid": "33293100",
  "title": "Rapid response to COVID-19, escalation and de-escalation strategies to match surge capacity of Intensive Care beds to a large scale epidemic.",
  "abstractText": "A major challenge during the COVID-19 outbreak is the sudden increase in ICU bed occupancy rate. In this article we reviewed the strategies of escalation and de-escalation put in place at a large university hospital in Madrid during the COVID-19 outbreak, in order to meet the growing demand of ICU beds.",
  "meshMajor": [
    "Beds",
    "COVID-19",
    "Epidemics",
    "Humans",
    "Intensive Care Units",
    "Spain",
    "Surge Capacity",
    "Time Factors"
  ],
  "labels": [
    "D001513",
    "D000086382",
    "D058872",
    "D006801",
    "D007362",
    "D013030",
    "D055872",
    "D013997"
  ],
  "journal": "Revista espanola de anestesiologia y reanimacion",
  "year": "2021",
  "redirection": "33293100",
  "candidate_labels": [
    [
      "Humans",
      1,
      1.0,
      1,
      0.9152058253524871,
      0,
      0,
      0,
      0,
      0.07937201365187713,
      0,
      0
    ],
    [
      "Surge Capacity",
      1,
      0.9999553858497469,
      1,
      0.36217614933085424,
      1,
      0,
      0,
      0,
      3.35e-07,
      1,
      0
    ],
    [
      "Critical Care",
      1,
      0.8914911175237251,
      1,
      0.32579419489726774,
      1,
      0,
      0,
      0,
      0.0017747440273037543,
      1,
      0
    ],
    [
      "COVID-19",
      1,
      0.8766252999400962,
      1,
      0.739179921067779,
      1,
      1,
      0,
      1,
      0.0009010238907849829,
      1,
      2
    ],
    [
      "Bed Occupancy",
      1,
      0.8463124563263813,
      1,
      0.1404667504730191,
      0,
      1,
      0,
      0,
      2.7303754266211605e-05,
      0,
      1
    ],
    [
      "Hospital Bed Capacity",
      1,
      0.8459607219342056,
      1,
      0.40559205338265186,
      0,
      0,
      0,
      0,
      5.460750853242321e-05,
      0,
      0
    ],
    [
      "SARS-CoV-2",
      1,
      0.790852554726851,
      1,
      0.45764436573537765,
      0,
      0,
      0,
      0,
      0.0007918088737201366,
      0,
      0
    ],
    [
      "Intensive Care Units",
      1,
      0.7847348444742954,
      1,
      0.6956371715385911,
      0,
      0,
      0,
      0,
      0.0010648464163822526,
      0,
      0
    ]
  ]
}

```

The description of the key '***candidate_labels***' is as follows:

*["SARS-CoV-2", 1, 0.790852554726851, 1, 0.45764436573537765, 0, 0, 0, 0, 0.0007918088737201366, 0, 0]*

**means**:

*[term_name, is_term_supported_by_MTI, MTI_normalized_score, is_term_supported_by_similarity, similarity_normalized_score, is_term_occur_in_title, is_term_in_abstract_first_sentence, is_term_in_abstract_middle_text, is_term_in_abstract_last_sentence, global_probability_of_term_occur_in_journal, term_frequency_in_title, term_frequency_in_abstract]*
