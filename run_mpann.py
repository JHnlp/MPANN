# -*- coding: utf-8 -*-
import argparse
import json
import pprint
import logging
import os
import collections
import numpy as np
import pathlib
import re
import csv
import torch
from functools import partial
from itertools import cycle, islice
from tqdm import tqdm
from torch.utils.data import DataLoader, IterableDataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam, ASGD, SGD, RMSprop, Adadelta, Adagrad, SparseAdam
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, StepLR, MultiStepLR
from tensorboardX import SummaryWriter
# from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertTokenizer
from transformers import (AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup)
from biosemin.processors.utils_glue import (convert_examples_to_features, processors,
                                            IterableProcessor, iterable_collate_fn, get_mlb)
from biosemin.models.multiprobe import MultiProbeNet
from biosemin.metrics.compute_metrics import compute_metrics
from biosemin.utils.common import seed_everything
from biosemin.utils.evaluation import EvalPointer, FlatEvaluator

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cudnn.benchmark = True
logger = logging.getLogger(__name__)

MODEL_TYPES = {
    'multiprobe': (BertConfig, MultiProbeNet, BertTokenizer),
}


class BioIterableDataset(IterableDataset):
    """ Iterable Dataset object is only used for large training dataset. """

    def __init__(self, file_path, start, end, step):
        self.file_path = file_path
        self.start = start if start >= 0 else 0
        self.end = end
        self.step = step

    def get_stream(self, file_path):
        with pathlib.Path(file_path).open('r') as file_obj:
            for ln in cycle(islice(file_obj, self.start, self.end, self.step)):  # 2,None,4
                yield ln.strip()

    def __iter__(self):
        return self.get_stream(self.file_path)


def load_and_cache_examples(args, tokenizer, mlb, data_type='train', iterable=False):
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training process the dataset,
        # and the others will use the cache
        torch.distributed.barrier()

    processor = IterableProcessor()
    # Load data features from cacheiter_dataset or dataset file
    if data_type == 'train':
        file_path = pathlib.Path(args.train_file)
        assert file_path.is_file(), '!!! Training file path error !!!'
    elif data_type == 'dev':
        file_path = pathlib.Path(args.dev_file)
        assert file_path.is_file(), '!!! Development file path error !!!'
    elif data_type == 'test':
        file_path = pathlib.Path(args.test_file)
        assert file_path.is_file(), '!!! Test file path error !!!'
    else:
        raise ValueError('!!! Specified wrong data type !!!')

    cached_features_file = os.path.join(args.data_dir,
                                        f'cached_{data_type}_{args.max_seq_length}_{file_path.stem}')
    print('')
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples = processor.get_examples(data_type=data_type, file_path=file_path)
        features = convert_examples_to_features(mlb=mlb,
                                                examples=examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length)

        if args.local_rank in [-1, 0]:  # torch.distributed.get_rank()
            logger.info("Saving features into cached file %s", cached_features_file)
            with pathlib.Path(cached_features_file).open('w', encoding='utf8') as f:
                for feature in features:
                    f.write(F'{feature}' + '\n')

    # if args.local_rank == 0 and not evaluate:
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    if iterable:
        return_dataset = BioIterableDataset(cached_features_file, start=args.local_rank, end=None,
                                            step=args.loading_interval)
    else:
        with pathlib.Path(cached_features_file).open('r', encoding='utf8') as f:
            return_dataset = [ln.strip() for ln in f]
    return return_dataset


def train(args, training_data, model, tokenizer, mlb, eval_data=None):
    """ Train the model """
    tb_writer = None
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu_per_process)
    # train_sampler = RandomSampler(training_data) if args.local_rank == -1 else DistributedSampler(training_data)
    train_dataloader = DataLoader(training_data,
                                  batch_size=args.train_batch_size,
                                  collate_fn=partial(iterable_collate_fn, mlb=mlb))
    if args.max_steps > 0:
        t_total = args.max_steps
    else:
        raise ValueError('Number of training max steps should greater than 0!')
    args.warmup_steps = int(t_total * args.warmup_proportion)
    logger.info("Total training step %s", t_total)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # if args.local_rank in [-1, 0]:
    #     pprint.pprint(optimizer_grouped_parameters)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    logger.info("Optimizer: %s", optimizer)
    logger.info("Scheduler: %s", scheduler)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu_per_process > 1:
        model = torch.nn.DataParallel(model)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    # Train!
    print('')
    logger.info("***** Running training *****")
    logger.info("===> Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("===> Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("===> Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("===> Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_loss = 1e8
    model.zero_grad()
    seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    with tqdm(train_dataloader, position=args.local_rank,
              desc='==>Training Rank: {}, Batch size: {}'.format(
                  args.local_rank, args.per_gpu_train_batch_size)) as pbar:
        for step, batch in enumerate(pbar):
            model.train()
            inputs = {'input_ids': batch['input_ids'].to(args.device),
                      'attention_mask': batch['attention_mask'].to(args.device),
                      'token_type_ids': batch['token_type_ids'].to(args.device),
                      'input_cand_label_token_ids': batch['cand_label_token_ids'].to(args.device),
                      'input_cand_label_token_length': batch['cand_label_token_length'].to('cpu'),
                      'input_cand_hit_mti': batch['cand_hit_mti'].to(args.device),
                      'input_cand_mti_probs': batch['cand_mti_probs'].to(args.device),
                      'input_cand_hit_neighbor': batch['cand_hit_neighbor'].to(args.device),
                      'input_cand_neighbor_probs': batch['cand_neighbor_probs'].to(args.device),
                      'input_cand_in_title': batch['cand_in_title'].to(args.device),
                      'input_cand_in_abf': batch['cand_in_abf'].to(args.device),
                      'input_cand_in_abm': batch['cand_in_abm'].to(args.device),
                      'input_cand_in_abl': batch['cand_in_abl'].to(args.device),
                      'input_cand_label_probs_in_jnl': batch['cand_label_probs_in_jnl'].to(args.device),
                      'input_cand_label_freq_in_title': batch['cand_label_freq_in_title'].to(args.device),
                      'input_cand_label_freq_in_ab': batch['cand_label_freq_in_ab'].to(args.device),
                      'input_jnl_token_ids': batch['jnl_token_ids'].to(args.device),
                      'input_jnl_token_length': batch['jnl_name_token_length'].to('cpu'),  # (batch_size,)
                      'target': batch['cand_label_match_gold_mask'].to(args.device)}
            batch_pmids = batch['guids'] if isinstance(batch['guids'], np.ndarray) \
                else batch['guids'].detach().cpu().numpy()
            outputs = model(**inputs)
            loss, logits = outputs[:2]  # model outputs are always tuple in pytorch-transformers (see doc)

            prediction = torch.sigmoid(logits).detach().cpu().numpy()
            target = inputs['target'].detach().cpu().numpy()  # candidate mesh descriptor mask for true labels

            if args.n_gpu_per_process > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if args.local_rank in [-1, 0]:
                tb_writer.add_scalar('batch_loss', loss.item(), step)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                # scheduler.step(loss.item())  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0]:
                    # tb_writer.add_scalar('stepped_lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('optimizer_lr', optimizer.param_groups[0]['lr'], global_step)

                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        keep_info = collections.OrderedDict()
                        pred_binary = np.where(prediction > 0.5, 1, 0)
                        keep_info['Train_All_target_sum'] = round(np.log2(target.sum() + args.epsilon), 6)
                        keep_info['Train_All_bin_pred_sum'] = round(np.log2(pred_binary.sum() + args.epsilon), 6)
                        keep_info['Train_All_logits_pred_sum'] = round(np.log2(prediction.sum() + args.epsilon), 6)
                        keep_info.update(compute_metrics(args.task_name, pred_binary, target))
                        for k, v in keep_info.items():
                            tb_writer.add_scalar(k, v, global_step)

                        average_loss = (tr_loss - logging_loss) / args.logging_steps
                        tb_writer.add_scalar('train_loss', average_loss, global_step)
                        logging_loss = tr_loss

                        if average_loss < best_loss:
                            best_loss = average_loss

                        logger.info("GPU Rank: %s, global_step: %s, train_loss: %1.5e, "
                                    "lr: %1.5e,  DocIDs: %s, Acc info: %s",
                                    args.local_rank, global_step, average_loss,
                                    optimizer.param_groups[0]['lr'],
                                    batch_pmids, keep_info)

                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        # Save model checkpoint
                        output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model,
                                                                'module') else model  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                        logger.info("Saving model checkpoint to %s", output_dir)
                        tokenizer.save_pretrained(save_directory=output_dir)

                        # evaluate
                        if args.local_rank == -1 and eval_data:  # Only evaluate when single GPU otherwise metrics may not average well
                            results = evaluate(args, eval_data, model, mlb,
                                               prefix="do_eval_checkpoint_" + str(global_step))

            # pbar(step, {'loss': loss.item(),
            #             'lr': scheduler.get_lr()[0]})
            pbar.set_postfix({'global step': '{}'.format(global_step),
                              'batch_loss': '{0:1.5e}'.format(loss.item()),
                              'lr': '{0:1.5e}'.format(optimizer.param_groups[0]['lr'])})

            if args.max_steps > 0 and global_step > args.max_steps:
                break
    if 'cuda' in str(args.device):
        torch.cuda.empty_cache()
    if args.local_rank in [-1, 0]:
        tb_writer.close()
    return global_step, tr_loss / global_step


def evaluate(args, eval_data, model, mlb, prefix=""):
    """ Don't use DP or DDP in Evaluation. """
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu_per_process)
    eval_dataloader = DataLoader(eval_data, sampler=SequentialSampler(eval_data),
                                 batch_size=args.eval_batch_size,
                                 collate_fn=partial(iterable_collate_fn, mlb=mlb))

    results = {}
    logger.info("***** Running evaluation on {} *****".format(prefix))
    logger.info("===>  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    all_pred_logits = None
    all_plausible_cand_labels = None
    all_gold_label_descriptors = []
    all_pmids = None
    all_cand_label_des_ids = None
    keep_info = collections.OrderedDict()
    with tqdm(eval_dataloader, desc="==>Evaluating, rank: {}".format(args.local_rank),
              position=args.local_rank + 1) as pbar:
        model.eval()
        for n_step, batch in enumerate(pbar):
            # convert sparse onehot vectors
            batch_true_label_descriptors = mlb.inverse_transform(batch['gold_label_des_onehot'])
            all_gold_label_descriptors.extend(batch_true_label_descriptors)

            # batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch['input_ids'].to(args.device),
                          'attention_mask': batch['attention_mask'].to(args.device),
                          'token_type_ids': batch['token_type_ids'].to(args.device),
                          'input_cand_label_token_ids': batch['cand_label_token_ids'].to(args.device),
                          'input_cand_label_token_length': batch['cand_label_token_length'].to('cpu'),
                          'input_cand_hit_mti': batch['cand_hit_mti'].to(args.device),
                          'input_cand_mti_probs': batch['cand_mti_probs'].to(args.device),
                          'input_cand_hit_neighbor': batch['cand_hit_neighbor'].to(args.device),
                          'input_cand_neighbor_probs': batch['cand_neighbor_probs'].to(args.device),
                          'input_cand_in_title': batch['cand_in_title'].to(args.device),
                          'input_cand_in_abf': batch['cand_in_abf'].to(args.device),
                          'input_cand_in_abm': batch['cand_in_abm'].to(args.device),
                          'input_cand_in_abl': batch['cand_in_abl'].to(args.device),
                          'input_cand_label_probs_in_jnl': batch['cand_label_probs_in_jnl'].to(args.device),
                          'input_cand_label_freq_in_title': batch['cand_label_freq_in_title'].to(args.device),
                          'input_cand_label_freq_in_ab': batch['cand_label_freq_in_ab'].to(args.device),
                          'input_jnl_token_ids': batch['jnl_token_ids'].to(args.device),
                          'input_jnl_token_length': batch['jnl_name_token_length'].to('cpu'),  # (batch_size,)
                          'target': batch['cand_label_match_gold_mask'].to(args.device)}
                batch_pmids = batch['guids'] if isinstance(batch['guids'], np.ndarray) \
                    else batch['guids'].detach().cpu().numpy()

                outputs = model(**inputs)
                batch_loss, logits = outputs[:2]
                # logger.info("batch loss: %s", batch_loss)
                eval_loss += batch_loss.mean().item()
                batch_pred_logits = torch.sigmoid(logits).detach().cpu().numpy()
                batch_plausible_gold_answer = inputs['target'].detach().cpu().numpy()

            nb_eval_steps += 1
            if all_pred_logits is None:
                all_pred_logits = batch_pred_logits
                all_plausible_cand_labels = batch_plausible_gold_answer
                all_pmids = batch_pmids
                all_cand_label_des_ids = batch['cand_label_des_ids']
            else:
                all_pred_logits = np.append(all_pred_logits, batch_pred_logits, axis=0)
                all_plausible_cand_labels = np.append(all_plausible_cand_labels, batch_plausible_gold_answer, axis=0)
                all_pmids = np.append(all_pmids, batch_pmids, axis=0)
                all_cand_label_des_ids = np.append(all_cand_label_des_ids, batch['cand_label_des_ids'], axis=0)
            pbar.set_postfix({'eval step': '{}'.format(n_step)})

    final_binary_prediction = np.where(all_pred_logits > 0.5, 1, 0)
    keep_info['Relaxed_eval_DocNum'] = len(all_pmids)
    keep_info['Relaxed_eval_All_target_sum'] = round(np.log2(all_plausible_cand_labels.sum() + args.epsilon), 6)
    keep_info['Relaxed_eval_All_bin_pred_sum'] = round(np.log2(final_binary_prediction.sum() + args.epsilon), 6)
    keep_info['Relaxed_eval_All_logits_pred_sum'] = round(np.log2(all_pred_logits.sum() + args.epsilon), 6)
    metrics_res = compute_metrics(args.task_name, final_binary_prediction, all_plausible_cand_labels)
    metrics_res = collections.OrderedDict(('Relaxed_' + k, v) for k, v in metrics_res.items())
    keep_info.update(metrics_res)

    eval_loss = eval_loss / nb_eval_steps
    results['eval_loss'] = eval_loss
    # results.update(metrics_res)
    results.update(keep_info)

    # logger.info("Evaluation info: %s", keep_info)
    assert len(all_pmids) == len(final_binary_prediction) == len(all_gold_label_descriptors)

    flateval = FlatEvaluator()
    gold_positive_answers, pred_positive_answers = collections.OrderedDict(), collections.OrderedDict()

    output_in_json = {}
    output_in_json['documents'] = []
    for i in range(len(all_pmids)):
        exam = {}
        exam['pmid'] = str(all_pmids[i])
        _cand_label_indicators = all_cand_label_des_ids[i][final_binary_prediction[i] == 1].tolist()
        exam['labels'] = [mlb.index_to_class[idx] for idx in _cand_label_indicators]
        output_in_json['documents'].append(exam)

        gold_positive_answers[exam['pmid']] = all_gold_label_descriptors[i]
        pred_positive_answers[exam['pmid']] = exam['labels']

    flateval.pred_data = pred_positive_answers
    flateval.gold_data = gold_positive_answers
    eval_performance = flateval.evaluate()
    eval_performance = collections.OrderedDict(('Stricted_' + k, v) for k, v in eval_performance.items())
    results.update(eval_performance)

    logger.info("Evaluation info: %s", results)

    file_stem = pathlib.Path(args.dev_file).stem
    fp_pred = pathlib.Path(args.output_dir).joinpath(file_stem + '_' + prefix + '.eval.json')
    with fp_pred.open('w', encoding='utf8') as eval_writer:
        json.dump(output_in_json, eval_writer, indent=2)

    if 'cuda' in str(args.device):
        torch.cuda.empty_cache()
    return results


def predict(args, eval_data, model, mlb, prefix=""):
    """ Don't use DP or DDP in prediction. """
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.pred_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu_per_process)
    pred_dataloader = DataLoader(eval_data, sampler=SequentialSampler(eval_data),
                                 batch_size=args.pred_batch_size,
                                 collate_fn=partial(iterable_collate_fn, mlb=mlb))

    results = {}
    logger.info("***** Running prediction on {} *****".format(prefix))
    logger.info("===>  Batch size = %d", args.pred_batch_size)
    nb_eval_steps = 0
    all_pred_logits = None
    all_pmids = None
    all_cand_label_des_ids = None
    with tqdm(pred_dataloader, desc="==>Evaluating, rank: {}".format(args.local_rank),
              position=args.local_rank + 1) as pbar:
        model.eval()
        for n_step, batch in enumerate(pbar):
            # batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch['input_ids'].to(args.device),
                          'attention_mask': batch['attention_mask'].to(args.device),
                          'token_type_ids': batch['token_type_ids'].to(args.device),
                          'input_cand_label_token_ids': batch['cand_label_token_ids'].to(args.device),
                          'input_cand_label_token_length': batch['cand_label_token_length'].to('cpu'),
                          'input_cand_hit_mti': batch['cand_hit_mti'].to(args.device),
                          'input_cand_mti_probs': batch['cand_mti_probs'].to(args.device),
                          'input_cand_hit_neighbor': batch['cand_hit_neighbor'].to(args.device),
                          'input_cand_neighbor_probs': batch['cand_neighbor_probs'].to(args.device),
                          'input_cand_in_title': batch['cand_in_title'].to(args.device),
                          'input_cand_in_abf': batch['cand_in_abf'].to(args.device),
                          'input_cand_in_abm': batch['cand_in_abm'].to(args.device),
                          'input_cand_in_abl': batch['cand_in_abl'].to(args.device),
                          'input_cand_label_probs_in_jnl': batch['cand_label_probs_in_jnl'].to(args.device),
                          'input_cand_label_freq_in_title': batch['cand_label_freq_in_title'].to(args.device),
                          'input_cand_label_freq_in_ab': batch['cand_label_freq_in_ab'].to(args.device),
                          'input_jnl_token_ids': batch['jnl_token_ids'].to(args.device),
                          'input_jnl_token_length': batch['jnl_name_token_length'].to('cpu'),  # (batch_size,)
                          'target': None}
                batch_pmids = batch['guids'] if isinstance(batch['guids'], np.ndarray) \
                    else batch['guids'].detach().cpu().numpy()

                outputs = model(**inputs)
                batch_loss, logits = outputs[:2]
                batch_pred_logits = torch.sigmoid(logits).detach().cpu().numpy()

            nb_eval_steps += 1
            if all_pred_logits is None:
                all_pred_logits = batch_pred_logits
                all_pmids = batch_pmids
                all_cand_label_des_ids = batch['cand_label_des_ids']
            else:
                all_pred_logits = np.append(all_pred_logits, batch_pred_logits, axis=0)
                all_pmids = np.append(all_pmids, batch_pmids, axis=0)
                all_cand_label_des_ids = np.append(all_cand_label_des_ids, batch['cand_label_des_ids'], axis=0)
            pbar.set_postfix({'pred step': '{}'.format(n_step)})

    final_binary_prediction = np.where(all_pred_logits > 0.5, 1, 0)
    assert len(all_pmids) == len(final_binary_prediction)

    output_in_json = {}
    output_in_json['documents'] = []
    for i in range(len(all_pmids)):
        exam = {}
        exam['pmid'] = str(all_pmids[i])
        _cand_label_indicators = all_cand_label_des_ids[i][final_binary_prediction[i] == 1].tolist()
        exam['labels'] = [mlb.index_to_class[idx] for idx in _cand_label_indicators]
        output_in_json['documents'].append(exam)

    file_stem = pathlib.Path(args.test_file).stem
    fp_pred = pathlib.Path(args.output_dir).joinpath(file_stem + '_' + prefix + '.pred.json')
    with fp_pred.open('w', encoding='utf8') as pred_writer:
        json.dump(output_in_json, pred_writer, indent=2)

    if 'cuda' in str(args.device):
        torch.cuda.empty_cache()
    return results


def parse_args():
    parser = argparse.ArgumentParser()

    # # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--mesh_file", default=None, type=str, required=True,
                        help="The input MeSH terms file.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_TYPES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # # Other parameters
    parser.add_argument("--config_name", default="bert_config.json", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run evaluation on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run prediction on the test set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
    parser.add_argument('--logging_steps', type=int, default=200,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=200,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--train_file', type=str, default=None, help="The training file path.")
    parser.add_argument('--dev_file', type=str, default=None, help="The development file path.")
    parser.add_argument('--test_file', type=str, default=None, help="The test file path.")
    parser.add_argument('--log_file', type=str, default=None, help="The log file path.")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # pprint.pprint(args.__dict__)  # print args parameters

    # rebase root directory
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:  # No GPU or Not use GPU
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")  # dp or cpu
        args.n_gpu_per_process = torch.cuda.device_count()  # Use DataParallel
        args.loading_interval = 1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)  # Use DistributedDataParallel
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        # torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
        args.n_gpu_per_process = 1
        args.loading_interval = torch.distributed.get_world_size()
    args.device = device

    # Setup logging
    print('*' * 50)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(args.log_file, mode='w', encoding='UTF-8')
    handler.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)
    logger.warning(
        "Process rank: %s, device: %s, n_gpu_per_process: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank, device, args.n_gpu_per_process, bool(args.local_rank != -1), args.fp16)

    # Set seed
    seed_everything(args.seed)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = IterableProcessor()
    label_collection = processor.get_labels(args.mesh_file)  # MeSH collection
    num_labels = len(label_collection.name_descriptor_mapping)  # number of MeSH collection

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    label_mlb = get_mlb(classes=label_collection.get_ranked_descriptor_names(), sparse_output=True)
    label_mlb.label_collection = label_collection

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_TYPES[args.model_type]
    logger.info("==> Task Model Type: {}".format(model_class))

    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config)
    # config = AutoConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    #                                           do_lower_case=args.do_lower_case)
    # model_2 = AutoModel.from_pretrained(args.model_name_or_path,
    #                                   from_tf=bool('.ckpt' in args.model_name_or_path),
    #                                   config=config)
    #
    # add new mark tokens
    num_of_words_to_add = tokenizer.add_tokens(['covid', 'coronavirus', 'sars'])
    num_of_special_to_add = tokenizer.add_special_tokens(
        {"additional_special_tokens": ['<TAG>', '<KEYWORD>', '<MESH>']})
    if num_of_special_to_add:  # add new embeddings or just pass
        model.resize_token_embeddings(tokenizer.vocab_size + num_of_words_to_add + num_of_special_to_add)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # assert args.optimizer in OPTIMIZER
    # if args.local_rank in [-1, 0]:
    #     print(' $ ' * 10)
    #     pprint.pprint(model.state_dict().keys())
    #     pprint.pprint(model.parameters())
    #     print(' $ ' * 10)

    model.to(args.device)
    print('*' * 50)
    logger.info("On GPU {}, Training/evaluation parameters {}".format(args.local_rank, args))

    train_dataset, eval_dataset, test_dataset = None, None, None
    # # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, label_mlb, data_type='train', iterable=True)
        eval_dataset = load_and_cache_examples(args, tokenizer, label_mlb, data_type='dev', iterable=False)

        global_step, tr_loss = train(args, training_data=train_dataset,
                                     model=model, tokenizer=tokenizer, mlb=label_mlb, eval_data=eval_dataset)
        logger.info("On GPU %s, global_step = %s, average loss = %s", args.local_rank, global_step, tr_loss)
        print('*' * 50)

        # Saving the last & best practices: if you use defaults names for the model, you can reload it using from_pretrained()
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            if global_step % args.save_steps != 0:
                output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                logger.info("Saving the last model checkpoint to %s", output_dir)
                # Save a trained model, configuration and tokenizer using `save_pretrained()`.
                # They can then be reloaded using `from_pretrained()`
                model_to_save = model.module if hasattr(model,
                                                        'module') else model  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                # Good practice: save your training arguments together with the trained model
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                # # Load a trained model and vocabulary that you have fine-tuned
                # model = model_class.from_pretrained(output_dir)
                # tokenizer = tokenizer_class.from_pretrained(output_dir, do_lower_case=args.do_lower_case)
                # model.to(args.device)

    print('*' * 50)
    # Evaluation
    if args.do_eval:
        eval_path = pathlib.Path(args.dev_file)
        if eval_path.is_file():
            eval_files = [eval_path]
        elif eval_path.is_dir():
            eval_files = sorted(list(eval_path.glob('*.jsonl')))
        else:
            raise ValueError('!!! Wrong path of evaluation dataset !!!')

        for evlf in eval_files:
            args.dev_file = str(evlf)
            print(' ==> evaluating {}'.format(args.dev_file))
            eval_dataset = load_and_cache_examples(args, tokenizer, label_mlb, data_type='dev', iterable=False)

            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = sorted(pathlib.Path(args.output_dir).glob('**/' + WEIGHTS_NAME))
                logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging

            if args.local_rank == -1:
                checkpoints_for_local_rank = checkpoints
                logger.info("Single mode evaluating, Local Rank -1.")
            else:
                world_size = torch.distributed.get_world_size()
                logger.info("Multi mode evaluating, World Size: %s, Local Rank: %s",
                            world_size, args.local_rank)

                checkpoints_for_local_rank = []
                for idx, checkpoint in enumerate(checkpoints):
                    if idx % world_size == args.local_rank:
                        checkpoints_for_local_rank.append(checkpoint)

            logger.info("Rank %s GPU is evaluating the following checkpoints: %s",
                        args.local_rank, checkpoints_for_local_rank)

            eval_results = collections.OrderedDict()
            columns = set()
            for checkpoint in checkpoints_for_local_rank:
                global_step = ''
                if 'checkpoint' in checkpoint.parent.name \
                        and len(checkpoint.parent.name.split('-')) > 1:
                    global_step = checkpoint.parent.name.split('-')[-1]

                config = config_class.from_pretrained(checkpoint.parent.joinpath('config.json'))
                model = model_class.from_pretrained(checkpoint, config=config)
                model.to(args.device)
                result = evaluate(args, eval_dataset, model, mlb=label_mlb, prefix=global_step)
                columns.update(result.keys())
                eval_results[str(checkpoint.parent.name)] = result

                if 'cuda' in str(args.device):
                    torch.cuda.empty_cache()
            columns = sorted(columns)
            write_rank_id = 'LocalRank_{}_'.format(args.local_rank) if args.local_rank >= 0 else 'Single_'
            with pathlib.Path(args.output_dir).joinpath(
                    '{}_eval_records.___.{}.___.csv'.format(write_rank_id, evlf.stem)
            ).open('w', encoding='utf8') as outf:
                tsv_w = csv.writer(outf, delimiter=',')
                tsv_w.writerow(['Entry'] + columns)
                for key, values in eval_results.items():
                    record = [key] + [values[col] for col in columns]
                    tsv_w.writerow(record)

    print('*' * 50)
    # Prediction
    if args.do_test:
        test_path = pathlib.Path(args.test_file)
        if test_path.is_file():
            test_files = [test_path]
        elif test_path.is_dir():
            test_files = sorted(list(test_path.glob('*.jsonl')))
        else:
            raise ValueError('!!! Wrong path of test dataset !!!')

        for testf in test_files:
            args.test_file = str(testf)
            print(' ==> predicting {}'.format(args.test_file))
            test_dataset = load_and_cache_examples(args, tokenizer, label_mlb, data_type='test', iterable=False)

            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = sorted(pathlib.Path(args.output_dir).glob('**/' + WEIGHTS_NAME))
                logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging

            if args.local_rank == -1:
                checkpoints_for_local_rank = checkpoints
                logger.info("Single mode evaluating, Local Rank -1.")
            else:
                world_size = torch.distributed.get_world_size()
                logger.info("Multi mode evaluating, World Size: %s, Local Rank: %s",
                            world_size, args.local_rank)

                checkpoints_for_local_rank = []
                for idx, checkpoint in enumerate(checkpoints):
                    if idx % world_size == args.local_rank:
                        checkpoints_for_local_rank.append(checkpoint)

            logger.info("Rank %s GPU is predicting the following checkpoints: %s",
                        args.local_rank, checkpoints_for_local_rank)

            for checkpoint in checkpoints_for_local_rank:
                global_step = ''
                if 'checkpoint' in checkpoint.parent.name \
                        and len(checkpoint.parent.name.split('-')) > 1:
                    global_step = checkpoint.parent.name.split('-')[-1]

                config = config_class.from_pretrained(checkpoint.parent.joinpath('config.json'))
                model = model_class.from_pretrained(checkpoint, config=config)
                model.to(args.device)
                result = predict(args, test_dataset, model, mlb=label_mlb, prefix=global_step)

                if 'cuda' in str(args.device):
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
    # processor = IterableProcessor()
    # label_list = processor.get_labels('data/covid19_mti/mesh_list.txt')
    # num_labels = len(label_list)
    # mlb = get_mlb(range(10))

    pass
