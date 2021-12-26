# -*- coding: utf-8 -*-
import csv
import logging
import os
import torch
import copy
import json
import pathlib
import numpy as np
from collections import OrderedDict, defaultdict, Counter
from dataclasses import dataclass, field
from typing import Optional
# from scipy.sparse import coo_matrix, csr_matrix, vstack, hstack
from io import open
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer

logger = logging.getLogger(__name__)

UNKOWN_LABEL = 'D000000'
UNKOWN_TERM = 'UNKOWN_TERM'


def clean_mesh_terms(term):
    """
        some MeSH terms contain '*' or '/', currently, we just filter these terms.
        i.g. Pneumonia, Viral / epidemiology*  -->  Pneumonia, Viral
            SARS Virus*  -->  SARS Virus
    """
    asterisk_pos = term.find('*')
    if asterisk_pos >= 0:
        term = term[:asterisk_pos].strip()
        slash_pos = term.find('/')
        if slash_pos >= 0:
            term = term[:slash_pos].strip()
    else:
        slash_pos = term.find('/')
        if slash_pos >= 0:
            term = term[:slash_pos].strip()
    return term


def get_mlb(classes=None, mlb=None, targets=None, sparse_output=False):
    if classes is not None:
        mlb = MultiLabelBinarizer(classes=classes, sparse_output=sparse_output)
        mlb.fit(None)

    if mlb is None and targets is not None:
        if isinstance(targets, csr_matrix):
            mlb = MultiLabelBinarizer(classes=range(targets.shape[1]), sparse_output=sparse_output)
            mlb.fit(None)
        else:
            mlb = MultiLabelBinarizer(sparse_output=sparse_output)
            mlb.fit(targets)

    mlb.class_to_index = dict(zip(mlb.classes_, range(len(mlb.classes_))))
    mlb.index_to_class = dict((v, k) for k, v in mlb.class_to_index.items())

    return mlb


def iterable_collate_fn(batch, mlb):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    batch_data = []
    for ln in batch:
        batch_data.append(eval(ln.strip()))

    batch_pmids, batch_input_length, \
    batch_input_ids, batch_token_type_ids, input_mask, \
    batch_unpadded_gold_label_des_ids, batch_cand_label_des_ids, \
    batch_cand_label_token_ids, batch_cand_label_token_length, batch_cand_label_match_gold_mask, \
    batch_cand_hit_mti, batch_cand_mti_probs, \
    batch_cand_hit_neighbor, batch_cand_neighbor_probs, \
    batch_cand_in_title, batch_cand_in_abf, batch_cand_in_abm, batch_cand_in_abl, \
    batch_cand_label_probs_in_jnl, \
    batch_cand_label_freq_in_title, batch_cand_label_freq_in_ab, \
    batch_jnl_token_ids, batch_jnl_name_token_length = zip(*batch_data)

    batch_size = len(batch)
    class_to_index = mlb.class_to_index

    # use numpy array for pmids
    batch_pmids = np.array(batch_pmids)  # str type

    # convert input tensor
    batch_input_length = torch.tensor(batch_input_length, dtype=torch.long)
    batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
    batch_attention_mask = torch.tensor(input_mask, dtype=torch.long)
    batch_token_type_ids = torch.tensor(batch_token_type_ids, dtype=torch.long)

    # 29000+ dimensions onehot vector for label
    batch_gold_label_onehot = torch.zeros(batch_size, len(class_to_index), dtype=torch.long)  # sparse label vector
    for i in range(batch_size):
        _unpadded_label_idx = torch.tensor(
            batch_unpadded_gold_label_des_ids[i], dtype=torch.long)
        batch_gold_label_onehot[i][_unpadded_label_idx] = 1

    batch_cand_label_des_ids = torch.tensor(batch_cand_label_des_ids, dtype=torch.long)
    batch_cand_label_token_ids = torch.tensor(batch_cand_label_token_ids, dtype=torch.long)
    batch_cand_label_token_length = torch.tensor(batch_cand_label_token_length, dtype=torch.long)
    batch_cand_label_match_gold_mask = torch.tensor(batch_cand_label_match_gold_mask, dtype=torch.long)

    batch_cand_hit_mti = torch.tensor(batch_cand_hit_mti, dtype=torch.long)
    batch_cand_mti_probs = torch.tensor(batch_cand_mti_probs, dtype=torch.float)
    batch_cand_hit_neighbor = torch.tensor(batch_cand_hit_neighbor, dtype=torch.long)
    batch_cand_neighbor_probs = torch.tensor(batch_cand_neighbor_probs, dtype=torch.float)

    batch_cand_in_title = torch.tensor(batch_cand_in_title, dtype=torch.long)
    batch_cand_in_abf = torch.tensor(batch_cand_in_abf, dtype=torch.long)
    batch_cand_in_abm = torch.tensor(batch_cand_in_abm, dtype=torch.long)
    batch_cand_in_abl = torch.tensor(batch_cand_in_abl, dtype=torch.long)

    batch_cand_label_probs_in_jnl = torch.tensor(batch_cand_label_probs_in_jnl, dtype=torch.float)
    batch_cand_label_freq_in_title = torch.tensor(batch_cand_label_freq_in_title, dtype=torch.long)
    batch_cand_label_freq_in_ab = torch.tensor(batch_cand_label_freq_in_ab, dtype=torch.long)
    batch_jnl_token_ids = torch.tensor(batch_jnl_token_ids, dtype=torch.long)
    batch_jnl_name_token_length = torch.tensor(batch_jnl_name_token_length, dtype=torch.long)

    batch_data = {'guids': batch_pmids,  # (batch_size,), str type
                  'input_ids': batch_input_ids,  # (batch_size, text_length)
                  'input_length': batch_input_length,  # (batch_size,)
                  'attention_mask': batch_attention_mask,  # (batch_size, text_length)
                  'token_type_ids': batch_token_type_ids,  # (batch_size, text_length)
                  'gold_label_des_onehot': batch_gold_label_onehot,  # (batch_size, 29917)
                  'cand_label_des_ids': batch_cand_label_des_ids,  # (batch_size, candidate_label_num)
                  'cand_label_match_gold_mask': batch_cand_label_match_gold_mask,  # (batch_size, candidate_label_num)
                  'cand_label_token_ids': batch_cand_label_token_ids,  # (batch_size, candidate_label_num, 30)
                  'cand_label_token_length': batch_cand_label_token_length,  # (batch_size, candidate_label_num)
                  'cand_hit_mti': batch_cand_hit_mti,  # (batch_size, candidate_label_num)
                  'cand_mti_probs': batch_cand_mti_probs,  # (batch_size, candidate_label_num)
                  'cand_hit_neighbor': batch_cand_hit_neighbor,  # (batch_size, candidate_label_num)
                  'cand_neighbor_probs': batch_cand_neighbor_probs,  # (batch_size, candidate_label_num)
                  'cand_in_title': batch_cand_in_title,  # (batch_size, candidate_label_num)
                  'cand_in_abf': batch_cand_in_abf,  # (batch_size, candidate_label_num)
                  'cand_in_abm': batch_cand_in_abm,  # (batch_size, candidate_label_num)
                  'cand_in_abl': batch_cand_in_abl,  # (batch_size, candidate_label_num)
                  'cand_label_probs_in_jnl': batch_cand_label_probs_in_jnl,  # (batch_size, candidate_label_num)
                  'cand_label_freq_in_title': batch_cand_label_freq_in_title,  # (batch_size, candidate_label_num)
                  'cand_label_freq_in_ab': batch_cand_label_freq_in_ab,  # (batch_size, candidate_label_num)
                  'jnl_token_ids': batch_jnl_token_ids,  # (batch_size, journal_length)
                  'jnl_name_token_length': batch_jnl_name_token_length  # (batch_size)
                  }

    return batch_data


@dataclass()
class InputExample(object):
    # __slots__ = ['pmid', 'text_a', 'text_b', 'labels', 'journal', 'candidate_labels']

    pmid: Optional[float] = field(default='')
    text_a: Optional[str] = field(default='')
    text_b: Optional[str] = field(default='')
    journal: Optional[str] = field(default='')
    labels: Optional[list] = field(default_factory=list)
    candidate_labels: Optional[list] = field(default_factory=list)

    def __post_init__(self):
        if not self.pmid:
            raise Exception('PMID should not be empty!')

    @property
    def text(self):
        return self.text_a + ' ' + self.text_b


@dataclass
class MeSHTerm:
    descriptor_UI: Optional[str] = field(default=None,
                                         metadata={"help": "The descriptor UI of MeSH terms should not be empty!"})
    descriptor_name: Optional[str] = field(default=None,
                                           metadata={"help": "The descriptor name of MeSH terms should not be empty!"})
    tree_number_list: Optional[list] = field(default_factory=list,
                                             metadata={"help": "List of tree numbers for a MeSH term."})
    aliases: Optional[list] = field(default_factory=list,
                                    metadata={"help": "List of multiple names for a MeSH term."})
    freq: Optional[float] = field(default=1.0,
                                  metadata={"help": "The frequency of a MeSH term."})
    global_prob: Optional[float] = field(default=1.0,
                                         metadata={"help": "The global likelyhood probability of a MeSH term."})
    tokens: Optional[list] = field(default_factory=list,
                                   metadata={"help": "The tokenized tokens inside a MeSH term."})

    def __post_init__(self):
        if not (self.descriptor_UI and self.descriptor_name):
            raise Exception(f'The MeSH term {self.descriptor_UI} for loading is in wrong format!')


class LabelVocab(object):
    def __init__(self, filepath):
        self.name_descriptor_mapping = OrderedDict()  # name string to descriptor
        self.id_descriptor_mapping = OrderedDict()  # descriptor to name string
        self._read_label_file(filepath)

    def __len__(self):
        return len(self.id_descriptor_mapping)

    def _read_label_file(self, filepath):
        with pathlib.Path(filepath).open("r", encoding="utf8") as f:
            for idx, ln in enumerate(f, 1):
                info = json.loads(ln.strip())
                mesh_term = MeSHTerm(descriptor_UI=info['descriptor_UI'],
                                     descriptor_name=info['descriptor_name'],
                                     tree_number_list=info['tree_number_list'],
                                     aliases=info['term_list'],
                                     freq=info['frequency'],
                                     global_prob=info['global_prob'],
                                     tokens=info['tokens'])
                self.name_descriptor_mapping[info['descriptor_name']] = mesh_term
                self.id_descriptor_mapping[info['descriptor_UI']] = mesh_term

    def get_descriptor_by_name(self, name):
        return self.name_descriptor_mapping.get(name, None)

    def get_descriptor_by_id(self, id):
        return self.id_descriptor_mapping.get(id, None)

    def get_ranked_descriptor_names(self, method='default'):
        if method.lower() == 'freq':
            lst = [(term_info.descriptor_name, term_info.freq)
                   for term_info in self.name_descriptor_mapping.values()]
            lst = sorted(lst, key=lambda x: x[-1], reverse=True)  # descending
            lst = [x[0] for x in lst]
        elif method.lower() == 'id':
            lst = [(term_info.descriptor_name, term_info.descriptor_UI)
                   for term_info in self.name_descriptor_mapping.values()]
            lst = sorted(lst, key=lambda x: x[-1])  # ascending
            lst = [x[0] for x in lst]
        else:
            lst = list(self.name_descriptor_mapping.keys())

        return lst


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self, data_dir):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_line(cls, input_file):
        with pathlib.Path(input_file).open('r', encoding='utf8') as reader:
            for ln in reader:
                yield ln.strip()


class IterableProcessor(DataProcessor):
    """Processor for the Relation data set."""

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        super(IterableProcessor, self).__init__()

    def _create_examples(self, json_lines):
        """Creates examples for the training and dev sets."""
        for (idx, ln) in enumerate(json_lines, 1):
            info = json.loads(ln)

            pmid = info.get("pmid", "")
            title = info.get("title", "")
            abstract = info.get("abstractText", "")
            journal = info.get("journal", '')
            # mesh terms deduplication
            mesh_major = info.get("meshMajor", [])  # Gold MeSH terms, perhaps labels of some cases would be empty
            # cleaned_mesh_terms = list(map(clean_mesh_terms, mesh_terms))  # temporary not used
            # labels = entry.get("labels", [])  # MeSH descriptors
            candidate_labels = info.get("candidate_labels", '')  # candidate mti descriptors

            # check the entry info
            if not pmid:
                print('Empty PMID!!!', ln)
                continue

            example = InputExample(pmid=pmid,
                                   text_a=title,
                                   text_b=abstract,
                                   journal=journal,
                                   labels=mesh_major,
                                   candidate_labels=candidate_labels)

            yield example

    def get_train_examples(self, path):
        """See base class."""
        path = pathlib.Path(path)
        if path.is_file():
            return self._create_examples(self._read_line(path))
        elif path.is_dir():
            return self._create_examples(
                self._read_line(os.path.join(path, "train.jsonl")))
        else:
            raise FileNotFoundError

    def get_dev_examples(self, path):
        """See base class."""
        path = pathlib.Path(path)
        if path.is_file():
            return self._create_examples(self._read_line(path))
        elif path.is_dir():
            return self._create_examples(
                self._read_line(os.path.join(path, "dev.jsonl")))
        else:
            raise FileNotFoundError

    def get_test_examples(self, path):
        """See base class."""
        path = pathlib.Path(path)
        if path.is_file():
            return self._create_examples(self._read_line(path))
        elif path.is_dir():
            return self._create_examples(
                self._read_line(os.path.join(path, "test.jsonl")))
        else:
            raise FileNotFoundError

    def get_examples(self, data_type, file_path):
        if data_type.lower() == 'train':
            return self.get_train_examples(file_path)
        elif data_type.lower() == 'dev':
            return self.get_dev_examples(file_path)
        else:
            return self.get_test_examples(file_path)

    def get_labels(self, path):
        """See base class."""
        path = pathlib.Path(path)
        if path.is_dir():
            label_voc = LabelVocab(pathlib.Path(path).joinpath("mesh_list.txt"))
            return label_voc
        elif path.is_file():
            label_voc = LabelVocab(path)
            return label_voc
        else:
            raise FileNotFoundError


def convert_examples_to_features(mlb, examples, tokenizer, max_seq_length,
                                 pad_token=0, pad_on_left=False,
                                 cls_token_at_end=False, cls_token='[CLS]', cls_token_segment_id=0,
                                 sep_token='[SEP]', pad_token_segment_id=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1, mask_padding_with_zero=True,
                                 max_jnl_seq_length=30, max_term_seq_length=30):
    '''
    logger.info("--Tokenize all mesh terms in advance, using bert tokenization")
    for des, text in mlb.dtos.items():
        mesh_tokens = tokenizer.encode(text,
                                       text_pair=None,
                                       add_special_tokens=False,
                                       max_length=max_term_length)
        all_mesh_term_token_indicators_dict[des] = mesh_tokens

        logger.info("--Tokenize all journal texts in advance, using bert tokenization")
        for jnl_name, jnl_no in journal_vocab.stod.items():
            jnl_tokens = tokenizer.encode(jnl_name,
                                          text_pair=None,
                                          add_special_tokens=False,
                                          max_length=max_journal_length)
            all_jnl_token_indicators_dict[int(jnl_no)] = jnl_tokens
    '''

    for (exam_idx, example) in enumerate(examples, 1):
        if exam_idx % 10000 == 0:
            logger.info("--Writing example %d " % (exam_idx))

        # journal
        jnl_tokens = tokenizer.tokenize(example.journal)
        jnl_name_ids = tokenizer.convert_tokens_to_ids(jnl_tokens)
        if len(jnl_name_ids) > max_jnl_seq_length:
            jnl_name_ids = jnl_name_ids[:max_jnl_seq_length]

        # true label
        _gold_labels_set = set(example.labels)  # for hash match
        unpadded_gold_label_ids = [mlb.class_to_index[lb] for lb in example.labels]

        # ---------------------------------------------------------------------------------
        # candidate labels
        # Candidiate Label Schema:
        #   e.g. ["Nanofibers", 0, 0.0, 1, 0.034446042067630356, 0, 0, 0, 0, 0.00016091609909437918, 0, 0]
        #       Term_name, Hit_by_MTI, MTI_Prob, Hit_by_Similarity, Similarity_Prob, \
        #       Is_in_Title, Is_in_Abstract_First_Sent, Is_in_Abstract_Middle_Text, Is_in_Abstract_Last_Sent, \
        #       Journal_Term_Prob, Term_Frequency_in_Title, Term_Frequency_in_Abstract
        cand_label_terms, cand_hit_mti, cand_mti_probs, cand_hit_neighbor, cand_neighbor_probs, \
        cand_in_title, cand_in_abf, cand_in_abm, cand_in_abl, \
        cand_label_probs_in_jnl, cand_label_freq_in_title, cand_label_freq_in_ab = list(zip(*example.candidate_labels))

        assert len(cand_label_terms) == 50

        cand_label_des_ids = [mlb.class_to_index[lb] for lb in cand_label_terms]
        cand_label_match_gold_mask = [1 if cand_term in _gold_labels_set else 0 for cand_term in cand_label_terms]
        cand_label_tokens = [tokenizer.tokenize(name) for name in cand_label_terms]  # list of list
        cand_label_token_ids = [tokenizer.convert_tokens_to_ids(tokens)
                                for tokens in cand_label_tokens]  # label term token indicators
        cand_label_token_ids = [token_ids[:max_term_seq_length] if len(token_ids) > max_term_seq_length else token_ids
                                for token_ids in cand_label_token_ids]

        # main text
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        tokens = tokens_a + [sep_token]
        token_type_indicators = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            left_space = max_seq_length - 2 - len(tokens)  # exclude '[CLS]' and last '[SEP]'
            if len(tokens_b) > left_space:
                pos = left_space // 2
                tokens_b = tokens_b[:pos] + tokens[-(left_space - pos):]

            tokens += tokens_b + [sep_token]
            token_type_indicators += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if len(tokens) > max_seq_length - 1:  # sequence is too long
            tokens = tokens[:max_seq_length - 1]
            tokens[-1] = sep_token
            token_type_indicators = token_type_indicators[:max_seq_length - 1]

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            token_type_indicators = token_type_indicators + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            token_type_indicators = [cls_token_segment_id] + token_type_indicators

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        input_text_len = len(input_ids)
        padding_length = max_seq_length - input_text_len
        jnl_padding_length = max_jnl_seq_length - len(jnl_name_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            token_type_indicators = ([pad_token_segment_id] * padding_length) + token_type_indicators

            jnl_name_token_length = len(jnl_name_ids)
            jnl_name_ids = [pad_token] * jnl_padding_length + jnl_name_ids
            cand_label_token_length = [len(ids) for ids in cand_label_token_ids]
            cand_label_token_ids = [[pad_token] * (max_term_seq_length - len(ids)) + ids
                                    for ids in cand_label_token_ids]
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_indicators = token_type_indicators + ([pad_token_segment_id] * padding_length)

            jnl_name_token_length = len(jnl_name_ids)
            jnl_name_ids = jnl_name_ids + [pad_token] * jnl_padding_length
            cand_label_token_length = [len(ids) for ids in cand_label_token_ids]
            cand_label_token_ids = [ids + [pad_token] * (max_term_seq_length - len(ids))
                                    for ids in cand_label_token_ids]

        assert len(input_ids) == max_seq_length, \
            "Error with input length {} vs {}".format(len(input_ids), max_seq_length)
        assert len(input_mask) == max_seq_length
        assert len(token_type_indicators) == max_seq_length

        if exam_idx < 3:
            print('\n')
            logger.info("*** Example ***")
            logger.info("pmid: %s" % (example.pmid))
            logger.info("input_indicators: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("token_type_indicators: %s" % " ".join([str(x) for x in token_type_indicators]))
            logger.info("labels num: %d" % (len(example.labels)))
            logger.info("candidate labels num: %d" % (len(example.candidate_labels)))
            logger.info("input length: %d" % (max_seq_length))

        exam_feature = [example.pmid, input_text_len,
                        input_ids, token_type_indicators, input_mask,
                        unpadded_gold_label_ids, cand_label_des_ids,
                        cand_label_token_ids, cand_label_token_length, cand_label_match_gold_mask,
                        cand_hit_mti, cand_mti_probs,
                        cand_hit_neighbor, cand_neighbor_probs,
                        cand_in_title, cand_in_abf, cand_in_abm, cand_in_abl,
                        cand_label_probs_in_jnl,
                        cand_label_freq_in_title, cand_label_freq_in_ab,
                        jnl_name_ids, jnl_name_token_length]

        yield exam_feature


processors = {
    "bioasq": IterableProcessor,
    "covid19": IterableProcessor,
}
if __name__ == '__main__':
    # processor = IterableProcessor()
    # tr = processor.get_train_examples('../../data')
    # label_list = processor.get_labels('../../data')

    pass
