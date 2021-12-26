import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import *
from transformers.models.bert import *
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


class MultiProbeNet(BertPreTrainedModel):
    def __init__(self, config):
        super(MultiProbeNet, self).__init__(config)
        self.num_labels = config.candidate_label_num
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        self.cand_name_encoder = nn.LSTM(input_size=config.hidden_size, hidden_size=config.rnn_hidden_size,
                                         num_layers=config.rnn_num_layers, batch_first=True, bidirectional=True)
        self.text_encoder = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size,
                                    num_layers=config.rnn_num_layers, batch_first=True, bidirectional=True)

        self._dynamic_probe_input = torch.arange(config.probe_num, dtype=torch.long)

        self.probe_embeddings_layer = nn.Embedding(config.probe_num, config.hidden_size, padding_idx=0)
        nn.init.xavier_uniform_(self.probe_embeddings_layer.weight)

        self.term_weighted_layer = nn.Linear(11 + 2 * config.rnn_num_layers * config.rnn_hidden_size,
                                             config.hidden_size)
        nn.init.xavier_uniform_(self.term_weighted_layer.weight)

        self.jnl_weighted_layer = nn.Linear(2 * config.rnn_num_layers * config.rnn_hidden_size, config.hidden_size)
        nn.init.xavier_uniform_(self.jnl_weighted_layer.weight)

        self.feature_compressed_layer = nn.Linear(9 * config.hidden_size, config.reduce_size)
        self.reLu = nn.LeakyReLU()
        self.model_classifier = nn.Linear(config.reduce_size, config.candidate_label_num)
        self.init_weights()

    def forward(self, input_ids,
                input_cand_label_token_ids, input_cand_label_token_length,
                input_cand_hit_mti, input_cand_mti_probs,
                input_cand_hit_neighbor, input_cand_neighbor_probs,
                input_cand_in_title, input_cand_in_abf, input_cand_in_abm, input_cand_in_abl,
                input_cand_label_probs_in_jnl, input_cand_label_freq_in_title, input_cand_label_freq_in_ab,
                input_jnl_token_ids, input_jnl_token_length,
                attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None,
                target=None):
        batch_size = input_ids.size(0)
        shared_bert_embedding_layer = self.bert.embeddings.word_embeddings  # reuse the bert embedding layer

        # 1. input text representation
        bert_output = self.bert(input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids,
                                head_mask=head_mask)
        pooled_output = bert_output[0]  # (batch_size, sequence_length, hidden_size)
        pooled_output = self.dropout(pooled_output)

        # 1.2 candidate label representation
        cand_label_token_embedding = shared_bert_embedding_layer(
            input_cand_label_token_ids)  # (batch_size, cand_num, token_num, hidden_size)
        cand_label_token_embedding = self.dropout(cand_label_token_embedding)

        # term encoding
        _, _cand_num, _padded_token_seq_length, _hidden_size = cand_label_token_embedding.size()
        # (batch_size * cand_num, token_num, hidden_size)
        _reshaped_cand_label_token_embedding = cand_label_token_embedding.view(-1, _padded_token_seq_length,
                                                                               _hidden_size)
        # (batch_size * cand_num)
        _reshaped_input_cand_label_token_length = input_cand_label_token_length.view(-1)
        padded_cand_label_token_embedding = pack_padded_sequence(_reshaped_cand_label_token_embedding,
                                                                 _reshaped_input_cand_label_token_length.cpu(),
                                                                 enforce_sorted=False, batch_first=True)
        term_encoded_outputs_packed, (term_h_last, term_c_last) = self.cand_name_encoder(
            padded_cand_label_token_embedding)
        term_encoded_outputs, _ = pad_packed_sequence(term_encoded_outputs_packed,
                                                      batch_first=True, total_length=_padded_token_seq_length)
        # (batch_size, cand_num, bidirection * rnn_num_layers * rnn_hidden_size)
        term_h_last = term_h_last.transpose(0, 1).contiguous().view(batch_size, _cand_num, -1)

        # 1.2.3 merge cand label representation
        cand_label_stat_repr = [input_cand_hit_mti.unsqueeze(-1),  # is term supported by MTI Online
                                input_cand_mti_probs.unsqueeze(-1),  # MTI Online normalized score
                                input_cand_hit_neighbor.unsqueeze(-1),  # is term supported by similarity
                                input_cand_neighbor_probs.unsqueeze(-1),  # similarity normalized score
                                input_cand_in_title.unsqueeze(-1),  # is term occurs in title
                                input_cand_in_abf.unsqueeze(-1),  # is term occurs in the first sentence of abstract
                                input_cand_in_abm.unsqueeze(-1),  # is term occurs in the middle of abstract
                                input_cand_in_abl.unsqueeze(-1),  # is term occurs in the last sentence of abstract
                                input_cand_label_probs_in_jnl.unsqueeze(-1),  # global prob of term occur in journal
                                input_cand_label_freq_in_title.unsqueeze(-1),  # term total freq in title
                                input_cand_label_freq_in_ab.unsqueeze(-1)]  # term total freq in abstract
        cand_label_stat_repr = torch.concat(cand_label_stat_repr, dim=-1)  # (batch_size, cand_num, 11)
        # normalization
        # (batch_size, cand_num, 11 + bidirection * rnn_num_layers * rnn_hidden_size)
        cand_label_repr = torch.concat([cand_label_stat_repr, term_h_last], dim=-1)
        # (batch_size, cand_num, hidden_size)
        cand_label_repr = self.term_weighted_layer(cand_label_repr)
        cand_label_repr = self.reLu(cand_label_repr)  # (batch_size, cand_num, hidden_size)
        kept_cand_label_repr = torch.mean(cand_label_repr, dim=1)  # (batch_size, hidden_size)

        # 1.3 journal embeddings
        jnl_padded_length = input_jnl_token_ids.size(-1)  # padded journal length
        # (batch_size, journal_length, hidden_size)
        jnl_token_embedding = shared_bert_embedding_layer(input_jnl_token_ids)
        jnl_token_embedding = self.dropout(jnl_token_embedding)
        padded_jnl_token_embedding = pack_padded_sequence(jnl_token_embedding,
                                                          input_jnl_token_length.cpu(),
                                                          enforce_sorted=False, batch_first=True)
        jnl_encoded_outputs_packed, (jnl_h_last, jnl_c_last) = self.cand_name_encoder(padded_jnl_token_embedding)
        # jnl_encoded_outputs size:(bidirection * rnn_num_layers, batch_size, rnn_hidden_size)
        jnl_encoded_outputs, _ = pad_packed_sequence(jnl_encoded_outputs_packed,
                                                     batch_first=True, total_length=jnl_padded_length)
        # (batch_size, bidirection * rnn_num_layers * rnn_hidden_size)
        jnl_h_last = jnl_h_last.transpose(0, 1).contiguous().view(batch_size, -1)
        jnl_repr = self.jnl_weighted_layer(jnl_h_last)  # (batch_size, hidden_size)
        jnl_repr = self.reLu(jnl_repr)  # (batch_size, hidden_size)

        # 1.4 dynamic probe embeddings
        probe_embedding = self.probe_embeddings_layer(
            self._dynamic_probe_input.to(input_ids.device))  # cpu or gpu
        probe_embedding = self.dropout(probe_embedding)  # (batch_size, probe_num, hidden_size)

        # -------------------------------------------------
        # 2. all attentions
        # 2.1 candidate_label-text attention
        cand_label_text_attn_logits = torch.matmul(
            pooled_output, cand_label_repr.transpose(-1, -2))  # (batch_size, sequence_length, cand_num)
        cand_label_text_attn_probs = F.softmax(cand_label_text_attn_logits, dim=-1)
        cand_label_text_attended_output = torch.matmul(
            cand_label_text_attn_probs, cand_label_repr)  # (batch_size, sequence_length, hidden_size)
        text_encoded_outputs_packed, (text_h_last, text_c_last) = self.text_encoder(cand_label_text_attended_output)
        # (batch_size, bidirection * rnn_num_layers * hidden_size)
        text_h_last = text_h_last.transpose(0, 1).contiguous().view(batch_size, -1)

        # 2.2 journal-cand_label attention
        cand_label_jnl_attn_logits = torch.matmul(
            jnl_repr.unsqueeze(1), cand_label_repr.transpose(-1, -2))  # (batch_size, 1, cand_num)
        cand_label_jnl_attn_probs = F.softmax(cand_label_jnl_attn_logits, dim=-1)  # (batch_size, 1, cand_num)
        cand_label_jnl_attn_probs = cand_label_jnl_attn_probs.transpose(-1, -2)  # (batch_size, cand_num, 1)
        cand_label_jnl_attended_output = cand_label_jnl_attn_probs * cand_label_repr  # (batch_size, cand_num, hidden_size)
        cand_label_jnl_attended_output = torch.sum(cand_label_jnl_attended_output, dim=1)  # (batch_size, hidden_size)

        # 2.3 journal-text attention
        jnl_text_attn_logits = torch.matmul(
            jnl_repr.unsqueeze(1), pooled_output.transpose(-1, -2))  # (batch_size, 1, sequence_length)
        jnl_text_attn_probs = F.softmax(jnl_text_attn_logits, dim=-1)  # (batch_size, 1, sequence_length)
        jnl_text_attn_probs = jnl_text_attn_probs.transpose(-1, -2)  # (batch_size, sequence_length, 1)
        jnl_text_attended_output = jnl_text_attn_probs * pooled_output  # (batch_size, sequence_length, hidden_size)
        jnl_text_attended_output = torch.sum(jnl_text_attended_output, dim=1)  # (batch_size, hidden_size)

        # 2.4 dynamic probe-text attention
        probe_text_attn_logits = torch.matmul(
            pooled_output, probe_embedding.transpose(-1, -2))  # (batch_size, sequence_length, probe_num)
        probe_text_attn_probs = F.softmax(probe_text_attn_logits, dim=-1)
        probe_text_attended_output = torch.matmul(
            probe_text_attn_probs, probe_embedding)  # (batch_size, sequence_length, hidden_size)
        probe_text_attended_output = torch.mean(probe_text_attended_output, dim=1)  # (batch_size, hidden_size)

        # 2.5 dynamic probe_journal attention
        probe_jnl_attn_logits = torch.matmul(
            jnl_repr.unsqueeze(1), probe_embedding.transpose(-1, -2))  # (batch_size, 1, probe_num)
        probe_jnl_attn_probs = F.softmax(probe_jnl_attn_logits, dim=-1)  # (batch_size, 1, probe_num)
        probe_jnl_attn_probs = probe_jnl_attn_probs.transpose(-1, -2)  # (batch_size, probe_num, 1)
        probe_jnl_attended_output = probe_jnl_attn_probs * probe_embedding  # (batch_size, probe_num, hidden_size)
        probe_jnl_attended_output = torch.mean(probe_jnl_attended_output, dim=1)  # (batch_size, hidden_size)

        # -------------------------------------------------
        # 3. final feature merging
        # 3.1 candidate label feature compress
        # [batch_size, (bidirection * rnn_num_layers + 2) * hidden_size]
        final_label_repr = [kept_cand_label_repr, text_h_last, cand_label_jnl_attended_output]
        final_label_repr = torch.concat(final_label_repr, dim=1)

        # 3.2 document feature compress
        # (batch_size, 3 * hidden_size)
        final_doc_repr = [jnl_text_attended_output, probe_text_attended_output, probe_jnl_attended_output]
        final_doc_repr = torch.concat(final_doc_repr, dim=1)

        # 3.3 feature fusion
        # (batch_size, 8 * hidden_size)
        merged_feature = [final_label_repr, final_doc_repr]
        merged_feature = torch.concat(merged_feature, dim=1)

        feature_compressed = self.feature_compressed_layer(merged_feature)  # (batch_size, 250)
        reduced_output = self.reLu(feature_compressed)

        # 4. classification
        logits = self.model_classifier(reduced_output)  # (batch_size, n_labels)
        loss = None
        if target is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, target.float())

        final_output = (loss, logits)
        return final_output  # (loss), logits, (hidden_states), (attentions)


if __name__ == '__main__':
    bilstm = nn.LSTM(input_size=10, hidden_size=2, num_layers=2, bidirectional=True)
    input = torch.randn(5, 3, 10)
    length = torch.tensor([5, 3, 4])

    h0 = torch.randn(4, 3, 2)
    c0 = torch.randn(4, 3, 2)
    output, (hn, cn) = bilstm(input, (h0, c0))

    embed_input_x_packed = pack_padded_sequence(input, length.cpu(), enforce_sorted=False, batch_first=False)
    encoder_outputs_packed, (h_last, c_last) = bilstm(embed_input_x_packed, (h0, c0))
    encoder_outputs, _ = pad_packed_sequence(encoder_outputs_packed, batch_first=False, total_length=5)

    print('output shape: ', output.shape)
    print('output: ', output[:, 1, :])
    print('encoder_outputs_packed shape: ', encoder_outputs[:, 1, :])

    print('hn shape: ', hn.shape)
    print('cn shape: ', cn.shape)

    print(output[4, 0, :2] == hn[2, 0])
    print(output[0, 0, 2:] == hn[3, 0])
    pass
