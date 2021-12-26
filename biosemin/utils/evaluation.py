# -*- coding: utf-8 -*-
"""

This script is originated from BioASQ Task A evaluation, for flat measures.

1. Before running the measures, the results of the system and the golden standard results need to be both complied with:
PMID1 D05632 D04322
PMID2 D033321 D98766 D98765
...

2. Run the flat measures the following command is invoked:
The program will print to the standard output the following numbers: accuracy EbP EbR EbF MaP MaR MaF MiP MiR MiF

3. For running the hierarchical measures: (not support currently)
./hierarchical/bin/HEMKit ./mesh/mesh_hier_int.txt true_labels_mapped.txt system_A_results_mapped.txt 4 5
will result to the following output: hP hR hF LCA-P LCA-R LCA-F
"""
import pathlib
import pprint
import json
from collections import OrderedDict

EPSILON = 1e-9


class EvalPointer(object):
    def __init__(self):
        self.tp = 0  # count for true positives
        self.tn = 0  # count for true negatives
        self.fp = 0  # count for false positives
        self.fn = 0  # count for false negatives

    def increase_tp(self):
        self.tp += 1

    def increase_tn(self):
        self.tn += 1

    def increase_fp(self):
        self.fp += 1

    def increase_fn(self):
        self.fn += 1


class FlatEvaluator(object):
    def __init__(self, gold_file=None, pred_file=None, epsilon=EPSILON):
        self.gold_file = gold_file
        self.pred_file = pred_file
        self.gold_data = None
        self.pred_data = None
        self.label_statistics = {}
        self.total_num_of_true_labels = 0
        self.total_num_of_pred_labels = 0
        self.epsilon = epsilon

    def _read_file(self, path, mode='flat', delimiter=' '):
        """
        flat format:
            PMID1 D05632 D04322
            PMID2 D033321 D98766 D98765

        json format:
            {"documents": [{"pmid": 28092139,"labels": ["D000328", "D000368", "D001132"]},
                    {"pmid": 28092888,"labels": ["D033321", "D98765", "D001132"]}]
            ...
        """
        if mode.lower() not in ['flat', 'json']:
            raise ValueError('Mode format error, mode not in ["flat", "json"]')

        file = pathlib.Path(path)

        if not file.is_file():
            print("File not found: " + path + " or unable to read file")
        else:
            content = {}

            if mode == 'flat':
                with file.open('r', encoding='utf8') as f:
                    for ln in f:
                        infos = ln.strip().split(delimiter)
                        content[infos[0]] = infos[1:]
            else:
                with file.open('r', encoding='utf8') as f:
                    docs = json.load(f)
                    for art in docs['documents']:
                        content[str(art['pmid'])] = art['labels']
            return content

    def increase_tp(self, class_id):
        if class_id in self.label_statistics:
            pointer = self.label_statistics[class_id]
            pointer.increase_tp()
        else:
            # print("Class id: %s" % class_id)
            self.label_statistics[class_id] = EvalPointer()
            pointer = self.label_statistics[class_id]
            pointer.increase_tp()

    def increase_tn(self, class_id):
        if class_id in self.label_statistics:
            pointer = self.label_statistics[class_id]
            pointer.increase_tn()
        else:
            # print("Class id: %s" % class_id)
            pointer = EvalPointer()
            self.label_statistics[class_id] = pointer
            pointer.increase_tn()

    def increase_fp(self, class_id):
        if class_id in self.label_statistics:
            pointer = self.label_statistics[class_id]
            pointer.increase_fp()
        else:
            # print("Class id: %s" % class_id)
            self.label_statistics[class_id] = EvalPointer()
            pointer = self.label_statistics[class_id]
            pointer.increase_fp()

    def increase_fn(self, class_id):
        if class_id in self.label_statistics:
            pointer = self.label_statistics[class_id]
            pointer.increase_fn()
        else:
            # print("Class id: %s" % class_id)
            self.label_statistics[class_id] = EvalPointer()
            pointer = self.label_statistics[class_id]
            pointer.increase_fn()

    def micro_precision(self):
        a = 0
        b = 0
        for key, pointer in self.label_statistics.items():
            a += pointer.tp
            b += pointer.tp + pointer.fp

        return float(a) / (float(b) + self.epsilon)

    def micro_recall(self):
        a = 0
        b = 0
        for key, pointer in self.label_statistics.items():
            a += pointer.tp
            b += pointer.tp + pointer.fn
        return float(a) / (float(b) + self.epsilon)

    def micro_f1(self):
        a = self.micro_precision()
        b = self.micro_recall()
        return 2.0 * a * b / (a + b + self.epsilon)

    def macro_precision(self):
        sum = 0.0
        for key, pointer in self.label_statistics.items():
            if pointer.tp == 0 and pointer.fp == 0:
                continue
            sum += float(pointer.tp) / float(pointer.tp + pointer.fp + self.epsilon)
        return 1.0 * sum / (self.total_num_of_pred_labels + self.epsilon)

    def macro_recall(self):
        sum = 0.0
        for key, pointer in self.label_statistics.items():
            if pointer.tp == 0 and pointer.fn == 0:
                continue
            sum += float(pointer.tp) / float(pointer.tp + pointer.fn + self.epsilon)
        return 1.0 * sum / (self.total_num_of_true_labels + self.epsilon)

    def macro_f1(self):
        pre = 0.0
        rec = 0.0
        macroF = 0.0

        for key, pointer in self.label_statistics.items():
            if pointer.tp != 0 or pointer.fp != 0:
                pre = float(pointer.tp) / float(pointer.tp + pointer.fp + self.epsilon)

            if pointer.tp != 0 or pointer.fn != 0:
                rec = float(pointer.tp) / float(pointer.tp + pointer.fn + self.epsilon)

            if pre != 0.0 or rec != 0.0:
                macroF += (2.0 * pre * rec) / (pre + rec + self.epsilon)

        return macroF / self.total_num_of_true_labels

    def evaluate(self, mode='flat', delimiter=' '):
        # load gold & pred data
        if self.gold_data is None:
            self.gold_data = self._read_file(self.gold_file, mode, delimiter)
        if self.pred_data is None:
            self.pred_data = self._read_file(self.pred_file, mode, delimiter)

        # prepare gold data
        for docid, gold_labels in self.gold_data.items():
            for lb in gold_labels:
                if lb not in self.label_statistics:
                    self.label_statistics[lb] = EvalPointer()
        self.total_num_of_true_labels = len(self.label_statistics)

        # evaluate
        label_type_accuracy = 0.0
        example_based_precision = 0.0
        example_based_recall = 0.0
        example_based_f1score = 0.0
        total_pred_labels = set()
        for pmid, pred_anws in self.pred_data.items():
            pred_labels = set(pred_anws)
            true_labels = set(self.gold_data[pmid])

            total_pred_labels.update(pred_labels)
            intersection_labels = true_labels & pred_labels

            for missed_label in true_labels - pred_labels:
                self.increase_fn(missed_label)
            for same_label in intersection_labels:
                self.increase_tp(same_label)
            for wrong_label in pred_labels - true_labels:
                self.increase_fp(wrong_label)

            label_type_accuracy += len(intersection_labels) / float(
                self.get_union_labels(true_labels, pred_labels) + self.epsilon)

            example_based_precision += 1.0 * len(intersection_labels) / (len(pred_labels) + self.epsilon)
            example_based_recall += 1.0 * len(intersection_labels) / float(len(true_labels) + self.epsilon)
            example_based_f1score += 2.0 * len(intersection_labels) / (
                    len(true_labels) + len(pred_labels) + self.epsilon)

        # for each test example
        self.total_num_of_pred_labels = len(total_pred_labels)
        eval_results = OrderedDict()
        eval_results['Label_Type_Accuracy'] = label_type_accuracy / float(len(self.pred_data) + self.epsilon)
        eval_results['EBP'] = example_based_precision / float(len(self.pred_data) + self.epsilon)
        eval_results['EBR'] = example_based_recall / float(len(self.pred_data) + self.epsilon)
        eval_results['EBF'] = example_based_f1score / float(len(self.pred_data) + self.epsilon)
        eval_results['MaP'] = self.macro_precision()
        eval_results['MaR'] = self.macro_recall()
        eval_results['MaF'] = self.macro_f1()
        eval_results['MiP'] = self.micro_precision()
        eval_results['MiR'] = self.micro_recall()
        eval_results['MiF'] = self.micro_f1()

        print()
        pprint.pprint(eval_results)

        return eval_results

    def get_union_labels(self, list1, list2):
        labels_per_instance = set()
        labels_per_instance.update(list1)
        labels_per_instance.update(list2)
        return len(labels_per_instance)

    def usage(self):
        """
        Describe parameters for calling the evaluation script
        """
        print("Usage: " + FlatEvaluator.__name__ + " goldendata systemanswers [-verbose]")
        print("goldendata systemanswers are the files (golden and submitted respectively)")
        print("verbose (optional) enables human readable output.")

    def isVerbosity(self):
        return self.verbosity

    def setVerbosity(self, verbosity):
        self.verbosity = verbosity


if __name__ == "__main__":
    pass
