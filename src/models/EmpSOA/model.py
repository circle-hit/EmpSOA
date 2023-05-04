### TAKEN FROM https://github.com/kolloldas/torchnlp
from multiprocessing.resource_sharer import stop
import os
import torch
import torch.nn as nn
from collections import Counter
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import math
from src.models.common import (
    EncoderLayer,
    DecoderLayer,
    LayerNorm,
    PositionwiseFeedForward,
    _gen_bias_mask,
    _gen_timing_signal,
    share_embedding,
    NoamOpt,
    _get_attn_subsequent_mask,
    get_input_from_batch,
    get_output_from_batch,
    top_k_top_p_filtering,
)
from src.utils import config
from src.utils.constants import MAP_EMO

from sklearn.metrics import accuracy_score


class Encoder(nn.Module):
    """
    A Transformer Encoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(
        self,
        embedding_size,
        hidden_size,
        num_layers,
        num_heads,
        total_key_depth,
        total_value_depth,
        filter_size,
        max_length=1000,
        input_dropout=0.0,
        layer_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        use_mask=False,
        universal=False,
    ):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """

        super(Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if self.universal:
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params = (
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            _gen_bias_mask(max_length) if use_mask else None,
            layer_dropout,
            attention_dropout,
            relu_dropout,
        )

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        if self.universal:
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(num_layers)])

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, mask):
        # Add input dropout
        x = self.input_dropout(inputs)

        # Project to hidden size
        x = self.embedding_proj(x)

        if self.universal:
            if config.act:
                x, (self.remainders, self.n_updates) = self.act_fn(
                    x,
                    inputs,
                    self.enc,
                    self.timing_signal,
                    self.position_signal,
                    self.num_layers,
                )
                y = self.layer_norm(x)
            else:
                for l in range(self.num_layers):
                    x += self.timing_signal[:, : inputs.shape[1], :].type_as(
                        inputs.data
                    )
                    x += (
                        self.position_signal[:, l, :]
                        .unsqueeze(1)
                        .repeat(1, inputs.shape[1], 1)
                        .type_as(inputs.data)
                    )
                    x = self.enc(x, mask=mask)
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)

            for i in range(self.num_layers):
                x = self.enc[i](x, mask)

            y = self.layer_norm(x)
        return y


class Decoder(nn.Module):
    """
    A Transformer Decoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(
        self,
        embedding_size,
        hidden_size,
        num_layers,
        num_heads,
        total_key_depth,
        total_value_depth,
        filter_size,
        max_length=1000,
        input_dropout=0.0,
        layer_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        universal=False,
    ):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(Decoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if self.universal:
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.mask = _get_attn_subsequent_mask(max_length)

        params = (
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            _gen_bias_mask(max_length),  # mandatory
            layer_dropout,
            attention_dropout,
            relu_dropout,
        )

        if self.universal:
            self.dec = DecoderLayer(*params)
        else:
            self.dec = nn.Sequential(
                *[DecoderLayer(*params) for l in range(num_layers)]
            )

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, encoder_output, mask):
        src_mask, mask_trg = mask
        dec_mask = torch.gt(
            mask_trg + self.mask[:, : mask_trg.size(-1), : mask_trg.size(-1)], 0
        )
        # Add input dropout
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)

        if self.universal:
            if config.act:
                x, attn_dist, (self.remainders, self.n_updates) = self.act_fn(
                    x,
                    inputs,
                    self.dec,
                    self.timing_signal,
                    self.position_signal,
                    self.num_layers,
                    encoder_output,
                    decoding=True,
                )
                y = self.layer_norm(x)

            else:
                x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)
                for l in range(self.num_layers):
                    x += (
                        self.position_signal[:, l, :]
                        .unsqueeze(1)
                        .repeat(1, inputs.shape[1], 1)
                        .type_as(inputs.data)
                    )
                    x, _, attn_dist, _ = self.dec(
                        (x, encoder_output, [], (src_mask, dec_mask))
                    )
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)

            # Run decoder
            y, _, attn_dist, _ = self.dec((x, encoder_output, [], (src_mask, dec_mask)))

            # Final layer normalization
            y = self.layer_norm(y)
        return y, attn_dist


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.p_gen_linear = nn.Linear(config.hidden_dim, 1)

    def forward(
        self,
        x,
        attn_dist=None,
        enc_batch_extend_vocab=None,
        extra_zeros=None,
        temp=1,
        beam_search=False,
        attn_dist_db=None,
    ):

        if config.pointer_gen:
            p_gen = self.p_gen_linear(x)
            alpha = torch.sigmoid(p_gen)

        logit = self.proj(x)

        if config.pointer_gen:
            vocab_dist = F.softmax(logit / temp, dim=2)
            vocab_dist_ = alpha * vocab_dist

            attn_dist = F.softmax(attn_dist / temp, dim=-1)
            attn_dist_ = (1 - alpha) * attn_dist
            enc_batch_extend_vocab_ = torch.cat(
                [enc_batch_extend_vocab.unsqueeze(1)] * x.size(1), 1
            )  ## extend for all seq
            if beam_search:
                enc_batch_extend_vocab_ = torch.cat(
                    [enc_batch_extend_vocab_[0].unsqueeze(0)] * x.size(0), 0
                )  ## extend for all seq
            logit = torch.log(
                vocab_dist_.scatter_add(2, enc_batch_extend_vocab_, attn_dist_)
            )
            return logit
        else:
            return F.log_softmax(logit, dim=-1)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        input_num = 4 if config.woEMO else 5 # 4 and 5
        input_dim = input_num * config.hidden_dim
        hid_num = 2 if config.woEMO else 3 # 2 and 3
        hid_dim = hid_num * config.hidden_dim
        out_dim = config.hidden_dim

        self.lin_1 = nn.Linear(input_dim, hid_dim, bias=False)
        self.lin_2 = nn.Linear(hid_dim, out_dim, bias=False)

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.lin_1(x)
        x = self.act(x)
        x = self.lin_2(x)

        return x

class MaskedSelfAttention(nn.Module):
    def __init__(self, config):
        super(MaskedSelfAttention, self).__init__()
        if config.hidden_dim % config.gat_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_dim, config.gat_heads))
        self.config = config
        self.num_attention_heads = config.gat_heads
        self.attention_head_size = int(config.hidden_dim / config.gat_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_dim, self.all_head_size)
        self.key = nn.Linear(config.hidden_dim, self.all_head_size)
        self.value = nn.Linear(config.hidden_dim, self.all_head_size)

        self.output = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.graph_layer_norm = LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(0.1)

        self.fusion = nn.Linear(2*config.hidden_dim, config.hidden_dim)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, role_mask, conv_len, state_type='user'):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        if role_mask is not None:
            role_mask = role_mask.unsqueeze(1).expand_as(attention_scores)
            attention_probs = nn.Softmax(dim=-1)(attention_scores.masked_fill(role_mask==0, -1e9))
        else:
            attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        context_layer = self.output(context_layer)
        context_layer = self.graph_layer_norm(context_layer + hidden_states)
        if not config.wo_dis_sel_oth:
            if config.dis_emo_cog:
                emo_states = []
                cog_states = []
                for item, idx in zip(context_layer, conv_len):
                    if state_type == 'user':
                        if not config.wo_csk:
                            cog_states.append(item[6*idx].unsqueeze(0))
                            emo_states.append(item[6*idx+1].unsqueeze(0))
                        else:
                            cog_states.append(item[idx].unsqueeze(0))
                            emo_states.append(item[idx+1].unsqueeze(0))
                    else:
                        if not config.wo_csk:
                            cog_states.append(item[6*idx+2].unsqueeze(0))
                            emo_states.append(item[6*idx+3].unsqueeze(0))
                        else:
                            cog_states.append(item[idx+2].unsqueeze(0))
                            emo_states.append(item[idx+3].unsqueeze(0))
                emo_states = torch.cat(emo_states, dim=0)
                cog_states = torch.cat(cog_states, dim=0)

                # states = emo_states + cog_states
                gate = nn.Sigmoid()(self.fusion(torch.cat([emo_states, cog_states], dim=-1)))
                states = gate * emo_states + (1-gate) * cog_states
                return context_layer, states, emo_states
            else:
                states = []
                for item, idx in zip(context_layer, conv_len):
                    if state_type == 'user':
                        states.append(item[6*idx].unsqueeze(0))
                    else:
                        states.append(item[6*idx+1].unsqueeze(0))

                states = torch.cat(states, dim=0)
                return context_layer, states, states
        else:
            emo_states = []
            cog_states = []
            for item, idx in zip(context_layer, conv_len):           
                cog_states.append(item[6*idx].unsqueeze(0))
                emo_states.append(item[6*idx+1].unsqueeze(0))             
            emo_states = torch.cat(emo_states, dim=0)
            cog_states = torch.cat(cog_states, dim=0)

            gate = nn.Sigmoid()(self.fusion(torch.cat([emo_states, cog_states], dim=-1)))
            states = gate * emo_states + (1-gate) * cog_states

            return context_layer, states, emo_states


class CrossAttention(nn.Module):
    def __init__(self, config):
        super(CrossAttention, self).__init__()
        if config.hidden_dim % config.gat_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_dim, config.gat_heads))
        self.config = config
        self.num_attention_heads = config.gat_heads
        self.attention_head_size = int(config.hidden_dim / config.gat_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_dim, self.all_head_size)
        self.key = nn.Linear(config.hidden_dim, self.all_head_size)
        self.value = nn.Linear(config.hidden_dim, self.all_head_size)

        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query_states, key_states, value_states, attention_mask=None):
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)

        # Normalize the attention scores to probabilities.
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).expand_as(attention_scores)
            attention_probs = nn.Softmax(dim=-1)(attention_scores.masked_fill(attention_mask==0, -1e9))
        else:
            attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

class EmpSOA(nn.Module):
    def __init__(
        self,
        vocab,
        decoder_number,
        model_file_path=None,
        is_eval=False,
        load_optim=False,
    ):
        super(EmpSOA, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words

        self.word_freq = np.zeros(self.vocab_size)

        self.is_eval = is_eval
        self.rels = ["x_intent", "x_need", "x_want", "x_effect", "x_react"] # 

        self.embedding = share_embedding(self.vocab, config.pretrain_emb)

        self.encoder = self.make_encoder(config.emb_dim)
        if config.csk_feature:
            self.emo_encoder = nn.Linear(config.csk_dim, config.hidden_dim)
            self.cog_encoder = nn.Linear(config.csk_dim, config.hidden_dim)
        
        if not config.wo_sod:
            if not config.wo_dis_sel_oth:
                if config.only_user:
                    self.self_other_interaction_1 = MaskedSelfAttention(config)
                elif config.only_agent:
                    self.self_other_interaction_2 = MaskedSelfAttention(config)
                else:
                    self.self_other_interaction_1 = MaskedSelfAttention(config)
                    self.self_other_interaction_2 = MaskedSelfAttention(config)
            else:
                self.graph_interaction = MaskedSelfAttention(config)

            if not config.wo_som:
                
                if config.only_user:
                    self.user_fusion = PositionwiseFeedForward(2*config.hidden_dim, 2*config.hidden_dim, config.hidden_dim)
                    self.ctx2user_cross_attention = CrossAttention(config)
                elif config.only_agent:
                    self.agent_fusion = PositionwiseFeedForward(2*config.hidden_dim, 2*config.hidden_dim, config.hidden_dim)
                    self.ctx2agent_cross_attention = CrossAttention(config)
                else:
                    self.user_fusion = PositionwiseFeedForward(2*config.hidden_dim, 2*config.hidden_dim, config.hidden_dim)
                    self.agent_fusion = PositionwiseFeedForward(2*config.hidden_dim, 2*config.hidden_dim, config.hidden_dim)

                    self.ctx2user_cross_attention = CrossAttention(config)
                    self.ctx2agent_cross_attention = CrossAttention(config)

                    self.reg_fusion = nn.Linear(2*config.hidden_dim, config.hidden_dim)

            if not config.wo_sog:
                if not config.only_user or not config.only_agent:
                    self.fusion = nn.Linear(3*config.hidden_dim, config.hidden_dim)

        self.decoder = Decoder(
            config.emb_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
        )

        self.emo_lin = nn.Linear(config.hidden_dim, decoder_number, bias=False)
        
        self.generator = Generator(config.hidden_dim, self.vocab_size)
        self.activation = nn.Softmax(dim=1)

        if config.weight_sharing:
            self.generator.proj.weight = self.embedding.lut.weight

        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx, reduction="sum")
        if not config.woDiv:
            self.criterion.weight = torch.ones(self.vocab_size)
        self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
        if config.noam:
            self.optimizer = NoamOpt(
                config.hidden_dim,
                1,
                8000,
                torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
            )

        if model_file_path is not None:
            print("loading weights")
            state = torch.load(model_file_path, map_location=config.device)
            self.load_state_dict(state["model"])
            if load_optim:
                self.optimizer.load_state_dict(state["optimizer"])
            self.eval()

        self.model_dir = config.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""

    def make_encoder(self, emb_dim):
        return Encoder(
            emb_dim,
            config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
            universal=config.universal,
        )

    def save_model(self, running_avg_ppl, iter):
        state = {
            "iter": iter,
            "optimizer": self.optimizer.state_dict(),
            "current_loss": running_avg_ppl,
            "model": self.state_dict(),
        }
        model_save_path = os.path.join(
            self.model_dir,
            "EmpSOA_{}_{:.4f}".format(iter, running_avg_ppl),
        )
        self.best_path = model_save_path
        torch.save(state, model_save_path)

    def clean_preds(self, preds):
        res = []
        preds = preds.cpu().tolist()
        for pred in preds:
            if config.EOS_idx in pred:
                ind = pred.index(config.EOS_idx) + 1  # end_idx included
                pred = pred[:ind]
            if len(pred) == 0:
                continue
            if pred[0] == config.SOS_idx:
                pred = pred[1:]
            res.append(pred)
        return res

    def update_frequency(self, preds):
        curr = Counter()
        for pred in preds:
            curr.update(pred)
        for k, v in curr.items():
            if k != config.EOS_idx:
                self.word_freq[k] += v

    def calc_weight(self):
        RF = self.word_freq / self.word_freq.sum()
        a = -1 / RF.max()
        weight = a * RF + 1
        weight = weight / weight.sum() * len(weight)

        return torch.FloatTensor(weight).to(config.device)

    def construct_graph(self, utt_emb, utt_cls_index, conv_len, cog_cls=None, emo_cls=None):
        aware_graph = []
        if not config.wo_dis_sel_oth:
            if config.dis_emo_cog:
                user_cog_state = torch.rand([utt_emb.size(0), config.emb_dim]).unsqueeze(1).type_as(utt_emb).cuda()
                agent_cog_state = torch.rand([utt_emb.size(0), config.emb_dim]).unsqueeze(1).type_as(utt_emb).cuda()
                user_emo_state = torch.rand([utt_emb.size(0), config.emb_dim]).unsqueeze(1).type_as(utt_emb).cuda()
                agent_emo_state = torch.rand([utt_emb.size(0), config.emb_dim]).unsqueeze(1).type_as(utt_emb).cuda()
            else:
                user_state = torch.rand([utt_emb.size(0), config.emb_dim]).unsqueeze(1).type_as(utt_emb).cuda()
                agent_state = torch.rand([utt_emb.size(0), config.emb_dim]).unsqueeze(1).type_as(utt_emb).cuda()
        else:
            emo_state = torch.rand([utt_emb.size(0), config.emb_dim]).unsqueeze(1).type_as(utt_emb).cuda()
            cog_state = torch.rand([utt_emb.size(0), config.emb_dim]).unsqueeze(1).type_as(utt_emb).cuda()
        
        utt_cls_embs =  []
        for item, idx in zip(utt_emb, utt_cls_index):
            cls_emb = torch.index_select(item, 0, idx)
            utt_cls_embs.append(cls_emb)
        utt_cls_embs = torch.stack(utt_cls_embs, dim=0)
    
        for i, cur_len in enumerate(conv_len):
            utt = utt_cls_embs[i, :cur_len]
            if not config.wo_csk:
                for j in range(cur_len):
                    for k in range(4): # number of cog csk
                        utt = torch.cat([utt, cog_cls[k][i][j].unsqueeze(0)], dim=0)
                    utt = torch.cat([utt, emo_cls[i][j].unsqueeze(0)], dim=0)
            
            if not config.wo_dis_sel_oth:
                if config.dis_emo_cog:
                    utt = torch.cat([utt, user_cog_state[i], user_emo_state[i], agent_cog_state[i], agent_emo_state[i]], dim=0)
                else:
                    utt = torch.cat([utt, user_state[i], agent_state[i]], dim=0)
            else:
                utt = torch.cat([utt, cog_state[i], emo_state[i]], dim=0)
            aware_graph.append(utt)
        aware_graph = pad_sequence(aware_graph, batch_first=True, padding_value=config.PAD_idx)
        return aware_graph

    def forward(self, batch):
        conv_len = [len(item) for item in batch["input_txt"]]
        enc_batch = batch["input_batch"]
        
        src_mask = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        user_seq_mask = (batch["mask_input"].data.eq(config.USR_idx).float().unsqueeze(1)) * (1-src_mask.float())
        agent_seq_mask = (batch["mask_input"].data.eq(config.SYS_idx).float().unsqueeze(1)) * (1-src_mask.float())
        
        mask_emb = self.embedding(batch["mask_input"]) # agent and user embedding
        src_emb = self.embedding(enc_batch) + mask_emb
        enc_outputs = self.encoder(src_emb, src_mask)  # batch_size * seq_len * 300
        utt_cls_index = batch["x_cls_index"]
        user_cls_index = batch["user_cls_index"]
        
        # Commonsense relations
        if not config.wo_csk:
            if config.csk_feature:
                cog_cls = []
                for r in self.rels[:-1]:
                    cog_cls.append(self.cog_encoder(batch[r]))
                emo_cls = self.emo_encoder(batch["x_react"])
            else:
                with torch.no_grad():
                    cs_embs = []
                    cs_masks = []
                    cs_outputs = {}
                    for r in self.rels:
                        csk_output = []
                        for csk in batch[r]:
                            emb = self.embedding(csk).to(config.device)
                            mask = csk.data.eq(config.PAD_idx).unsqueeze(1)
                            cs_embs.append(emb)
                            cs_masks.append(mask)
                            if r != "x_react":
                                enc_output = self.encoder(emb, mask)[:, 0] # get [CLS] token
                            else:
                                enc_output = torch.mean(self.encoder(emb, mask), dim=1) # avg
                            csk_output.append(enc_output)
                        cs_outputs[r] = pad_sequence(csk_output, batch_first=True, padding_value=config.PAD_idx)

                # Shape: batch_size * 1 * 300
                cog_cls = []
                for r in self.rels[:-1]:
                    cog_cls.append(cs_outputs[r])
                emo_cls = cs_outputs["x_react"]
        
        # -----------------SOD-----------------
        if not config.wo_sod:
            if not config.wo_csk:
                self_other_graph = self.construct_graph(src_emb, utt_cls_index, conv_len, cog_cls, emo_cls)
            else:
                self_other_graph = self.construct_graph(src_emb, utt_cls_index, conv_len)
            
            if not config.wo_dis_sel_oth:
                if config.only_user:
                    x, user_state, user_emo_state = self.self_other_interaction_1(self_other_graph, batch["user_mask"], conv_len, 'user')
                    x_user, user_state, user_emo_state = self.self_other_interaction_1(x, batch["user_mask"], conv_len, 'user')
                    agent_state = None
                elif config.only_agent:
                    x, agent_state, agent_emo_state = self.self_other_interaction_2(self_other_graph, batch["agent_mask"], conv_len, 'agent')
                    x_agent, agent_state, agent_emo_state = self.self_other_interaction_2(x, batch["agent_mask"], conv_len, 'agent')
                    user_state = None
                else:
                    x, user_state, user_emo_state = self.self_other_interaction_1(self_other_graph, batch["user_mask"], conv_len, 'user')
                    x_user, user_state, user_emo_state = self.self_other_interaction_1(x, batch["user_mask"], conv_len, 'user')

                    x, agent_state, agent_emo_state = self.self_other_interaction_2(self_other_graph, batch["agent_mask"], conv_len, 'agent')
                    x_agent, agent_state, agent_emo_state = self.self_other_interaction_2(x, batch["agent_mask"], conv_len, 'agent')
            
            else:
                x, user_state, user_emo_state = self.graph_interaction(self_other_graph, batch["graph_mask"], conv_len)
                x, user_state, user_emo_state = self.graph_interaction(x, batch["graph_mask"], conv_len)
                
                agent_state = user_state
        else:
            agent_state, user_state = None, None
        # -----------------------------------------------------------

        # --------------------Emotion Perception---------------------
        user_cls_embs =  []
        for item, idx, cur_len in zip(enc_outputs, user_cls_index, conv_len):
            cls_emb = torch.index_select(item, 0, idx)
            cur_user_len = (cur_len + 1) // 2
            usr_emb = cls_emb[:cur_user_len][:]
            cur_mean = torch.mean(usr_emb, dim=0)
            user_cls_embs.append(cur_mean)
        user_cls_embs = torch.stack(user_cls_embs, dim=0)
        
        if not config.wo_sod:
            if config.only_agent:
                emo_logits = self.emo_lin(agent_emo_state + user_cls_embs)
            else:
                emo_logits = self.emo_lin(user_emo_state + user_cls_embs)
        else:
            emo_logits = self.emo_lin(user_cls_embs)
        # -----------------------------------------------------------

        # -----------------SOM-----------------
        if not config.wo_som:
            if config.only_user:
                user_ctx = self.user_fusion(torch.cat([enc_outputs, user_state.unsqueeze(1).expand_as(enc_outputs)], dim=-1))
                user_ref_ctx = self.ctx2user_cross_attention(enc_outputs, user_ctx, user_ctx, user_seq_mask)
                out_ctx = user_ref_ctx
            elif config.only_agent:
                agent_ctx = self.agent_fusion(torch.cat([enc_outputs, agent_state.unsqueeze(1).expand_as(enc_outputs)], dim=-1))          
                agent_ref_ctx = self.ctx2agent_cross_attention(enc_outputs, agent_ctx, agent_ctx, agent_seq_mask)
                out_ctx = agent_ref_ctx
            else:
                user_ctx = self.user_fusion(torch.cat([enc_outputs, user_state.unsqueeze(1).expand_as(enc_outputs)], dim=-1))
                agent_ctx = self.agent_fusion(torch.cat([enc_outputs, agent_state.unsqueeze(1).expand_as(enc_outputs)], dim=-1))   

                user_ref_ctx = self.ctx2user_cross_attention(enc_outputs, user_ctx, user_ctx, user_seq_mask)
                agent_ref_ctx = self.ctx2agent_cross_attention(enc_outputs, agent_ctx, agent_ctx, agent_seq_mask)

                gate = nn.Sigmoid()(self.reg_fusion(torch.cat([user_ref_ctx, agent_ref_ctx], dim=-1)))
                out_ctx = gate * user_ref_ctx + (1-gate) * agent_ref_ctx
        else:
            out_ctx = enc_outputs
        
        # -----------------------------------------------------------
        
        return src_mask, out_ctx, emo_logits, user_state, agent_state

    def train_one_batch(self, batch, iter, train=True):
        (
            enc_batch,
            _,
            _,
            enc_batch_extend_vocab,
            extra_zeros,
            _,
            _,
            _,
        ) = get_input_from_batch(batch)
        dec_batch, _, _, _, _ = get_output_from_batch(batch)

        if config.noam:
            self.optimizer.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()

        src_mask, ctx_output, emo_logits, user_state, agent_state = self.forward(batch)

        # Decode
        sos_token = (
            torch.LongTensor([config.SOS_idx] * enc_batch.size(0))
            .unsqueeze(1)
            .to(config.device)
        )
        dec_batch_shift = torch.cat((sos_token, dec_batch[:, :-1]), dim=1)
        mask_trg = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)

        # batch_size * seq_len * 300 (GloVe)
        dec_emb = self.embedding(dec_batch_shift)
        
        pre_logit, attn_dist = self.decoder(dec_emb, ctx_output, (src_mask, mask_trg))
        
        #----------------SOG-----------------
        if not config.wo_sog:
            if config.only_user:
                pre_logit = pre_logit + user_state.unsqueeze(1).expand_as(pre_logit)
            elif config.only_agent:
                pre_logit = pre_logit + agent_state.unsqueeze(1).expand_as(pre_logit)
            else:
                gate = nn.Sigmoid()(self.fusion(torch.cat([pre_logit, user_state.unsqueeze(1).expand_as(pre_logit), agent_state.unsqueeze(1).expand_as(pre_logit)], dim=-1)))
                pre_logit = pre_logit + gate * user_state.unsqueeze(1).expand_as(pre_logit) + (1-gate) * agent_state.unsqueeze(1).expand_as(pre_logit)

        ## compute output dist
        logit = self.generator(
            pre_logit,
            attn_dist,
            enc_batch_extend_vocab if config.pointer_gen else None,
            extra_zeros,
            attn_dist_db=None,
        )

        emo_label = torch.LongTensor(batch["program_label"]).to(config.device)
        emo_loss = nn.CrossEntropyLoss()(emo_logits, emo_label).to(config.device)
        ctx_loss = self.criterion_ppl(
            logit.contiguous().view(-1, logit.size(-1)),
            dec_batch.contiguous().view(-1),
        )

        if not (config.woDiv):
            _, preds = logit.max(dim=-1)
            preds = self.clean_preds(preds)
            self.update_frequency(preds)
            self.criterion.weight = self.calc_weight()
            not_pad = dec_batch.ne(config.PAD_idx)
            target_tokens = not_pad.long().sum().item()
            div_loss = self.criterion(
                logit.contiguous().view(-1, logit.size(-1)),
                dec_batch.contiguous().view(-1),
            )
            div_loss /= target_tokens
            loss = 1.5 * div_loss + ctx_loss + emo_loss
        else:
            loss = ctx_loss + emo_loss

        pred_program = np.argmax(emo_logits.detach().cpu().numpy(), axis=1)
        program_acc = accuracy_score(batch["program_label"], pred_program)

        # print results for testing
        top_preds = ""
        comet_res = {}

        if self.is_eval:
            top_preds = emo_logits.detach().cpu().numpy().argsort()[0][-3:][::-1]
            top_preds = f"{', '.join([MAP_EMO[pred.item()] for pred in top_preds])}"
            if not config.csk_feature:
                for r in self.rels:
                    txt = [[[" ".join(t) for t in tn] for tn in tm] for tm in batch[f"{r}_txt"]]
                    comet_res[r] = txt

        if train:
            loss.backward()
            self.optimizer.step()

        return (
            ctx_loss.item(),
            math.exp(min(ctx_loss.item(), 100)),
            emo_loss.item(),
            program_acc,
            top_preds,
            comet_res,
        )

    def compute_act_loss(self, module):
        R_t = module.remainders
        N_t = module.n_updates
        p_t = R_t + N_t
        avg_p_t = torch.sum(torch.sum(p_t, dim=1) / p_t.size(1)) / p_t.size(0)
        loss = config.act_loss_weight * avg_p_t.item()
        return loss

    def decoder_greedy(self, batch, max_dec_step=30):
        (
            _,
            _,
            _,
            enc_batch_extend_vocab,
            extra_zeros,
            _,
            _,
            _,
        ) = get_input_from_batch(batch)
        src_mask, ctx_output, _, user_state, agent_state = self.forward(batch)

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long().to(config.device)

        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            ys_embed = self.embedding(ys)
            if config.project:
                out, attn_dist = self.decoder(
                    self.embedding_proj_in(ys_embed),
                    self.embedding_proj_in(ctx_output),
                    (src_mask, mask_trg),
                )
            else:
                out, attn_dist = self.decoder(
                    ys_embed, ctx_output, (src_mask, mask_trg)
                )
            
            if not config.wo_sog:
                if config.only_user:
                    out = out + user_state.unsqueeze(1).expand_as(out)
                elif config.only_agent:
                    out = out + agent_state.unsqueeze(1).expand_as(out)
                else:
                    gate = nn.Sigmoid()(self.fusion(torch.cat([out, user_state.unsqueeze(1).expand_as(out), agent_state.unsqueeze(1).expand_as(out)], dim=-1)))
                    out = out + gate * user_state.unsqueeze(1).expand_as(out) + (1-gate) * agent_state.unsqueeze(1).expand_as(out)

            prob = self.generator(
                out, attn_dist, enc_batch_extend_vocab, extra_zeros, attn_dist_db=None
            )
            _, next_word = torch.max(prob[:, -1], dim=1)
            decoded_words.append(
                [
                    "<EOS>"
                    if ni.item() == config.EOS_idx
                    else self.vocab.index2word[ni.item()]
                    for ni in next_word.view(-1)
                ]
            )
            next_word = next_word.data[0]

            ys = torch.cat(
                [ys, torch.ones(1, 1).long().fill_(next_word).to(config.device)],
                dim=1,
            ).to(config.device)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ""
            for e in row:
                if e == "<EOS>":
                    break
                else:
                    st += e + " "
            sent.append(st)
        return sent

    def decoder_topk(self, batch, max_dec_step=30):
        (
            enc_batch,
            _,
            _,
            enc_batch_extend_vocab,
            extra_zeros,
            _,
            _,
            _,
        ) = get_input_from_batch(batch)
        src_mask, ctx_output, _ = self.forward(batch)

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long().to(config.device)
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            if config.project:
                out, attn_dist = self.decoder(
                    self.embedding_proj_in(self.embedding(ys)),
                    self.embedding_proj_in(ctx_output),
                    (src_mask, mask_trg),
                )
            else:
                out, attn_dist = self.decoder(
                    self.embedding(ys), ctx_output, (src_mask, mask_trg)
                )

            logit = self.generator(
                out, attn_dist, enc_batch_extend_vocab, extra_zeros, attn_dist_db=None
            )
            filtered_logit = top_k_top_p_filtering(
                logit[0, -1] / 0.7, top_k=0, top_p=0.9, filter_value=-float("Inf")
            )
            # Sample from the filtered distribution
            probs = F.softmax(filtered_logit, dim=-1)

            next_word = torch.multinomial(probs, 1).squeeze()
            decoded_words.append(
                [
                    "<EOS>"
                    if ni.item() == config.EOS_idx
                    else self.vocab.index2word[ni.item()]
                    for ni in next_word.view(-1)
                ]
            )
            # _, next_word = torch.max(logit[:, -1], dim=1)
            next_word = next_word.item()

            ys = torch.cat(
                [ys, torch.ones(1, 1).long().fill_(next_word).to(config.device)],
                dim=1,
            ).to(config.device)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ""
            for e in row:
                if e == "<EOS>":
                    break
                else:
                    st += e + " "
            sent.append(st)
        return sent