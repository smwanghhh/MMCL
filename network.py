import math
import os
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from einops import rearrange, repeat
from torch.nn import functional as F


class Self_Attention(nn.Module):
    def __init__(self, emb_size, head, dropout):
        super(Self_Attention, self).__init__()

        self.num_attention_heads = head
        self.attention_head_size = emb_size // head
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dropout = nn.Dropout(dropout)

        self.query = nn.Linear(emb_size, self.all_head_size)
        self.key = nn.Linear(emb_size, self.all_head_size)
        self.value = nn.Linear(emb_size, self.all_head_size)

        self.dense1 = nn.Linear(emb_size, emb_size)
        self.norm = nn.LayerNorm(emb_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input):
        mixed_query_layer = self.query(input)  # [Batch_size x Seq_length x Hidden_size]
        mixed_key_layer = self.key(input)  # [Batch_size x Seq_length x Hidden_size]
        mixed_value_layer = self.value(input)  # [Batch_size x Seq_length x Hidden_size]

        query_layer = self.transpose_for_scores(
            mixed_query_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        value_layer = self.transpose_for_scores(
            mixed_value_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        attention_probs = self.dropout(nn.Softmax(dim=-1)(attention_scores))  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        context_layer = torch.matmul(attention_probs, value_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]

        context_layer = context_layer.permute(0, 2, 1,
                                              3).contiguous()  # [Batch_size x Seq_length x Num_of_heads x Head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size,)  # [Batch_size x Seq_length x Hidden_size]
        context_layer = context_layer.view(*new_context_layer_shape)  # [Batch_size x Seq_length x Hidden_size]
        out = context_layer + self.dense1(context_layer)
  
        return self.norm(out)
    
class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.d_t, self.d_a, self.d_v, self.d_target = args.dim_t, args.dim_a, args.dim_v, args.emb_size
        self.att_dp = args.atten_dropout
        self.n_head = args.n_head
        self.text_map = nn.Linear(self.d_t, self.d_target)
        self.audio_map = nn.Linear(self.d_a, self.d_target)
        self.vision_map = nn.Linear(self.d_v, self.d_target)
        self.attention_text = Self_Attention(self.d_target, self.n_head, self.att_dp)
        self.attention_audio = Self_Attention(self.d_target, self.n_head, self.att_dp)
        self.attention_vision = Self_Attention(self.d_target, self.n_head, self.att_dp)


    def map_to_subspace(self, text, audio, vision): 
        Similarity_T, Similarity_A, Similarity_V = self.similarity_matrix(text, audio, vision)

        shared_text, shared_audio, shared_vision = torch.matmul(Similarity_T, text), torch.matmul(Similarity_A, audio), torch.matmul(Similarity_V, vision)
        different_text, different_audio, different_vision = torch.matmul(nn.Softmax(-1)(1 - Similarity_T), text), torch.matmul(nn.Softmax(-1)(1 - Similarity_A), audio), \
                                                            torch.matmul(nn.Softmax(-1)(1 - Similarity_V), vision)
        shared_text = self.attention_text(shared_text)
        shared_audio = self.attention_audio(shared_audio)
        shared_vision = self.attention_vision(shared_vision)
        shared_emb = torch.cat((shared_text, shared_audio, shared_vision), 1)
        diff_emb = torch.cat((different_text, different_audio, different_vision), 1) 
        return shared_emb, diff_emb      



    def similarity_matrix(self, emb_t, emb_a, emb_v):
        emb_t_norm = F.normalize(emb_t, p=2, dim = 2)
        emb_a_norm = F.normalize(emb_a, p=2, dim = 2)
        emb_v_norm = F.normalize(emb_v, p=2, dim = 2)

        ###text TA, TV
        similarity_ta = torch.matmul(emb_t_norm, emb_a_norm.permute(0, 2, 1))
        similarity_tv = torch.matmul(emb_t_norm, emb_v_norm.permute(0, 2, 1))
        # Similarity_T = (similarity_ta + similarity_tv) / 2
        Similarity_T = nn.Softmax(-1)(torch.min(similarity_ta, similarity_tv))        
        ###audio, AT, AV
        similarity_at = torch.matmul(emb_a_norm,emb_t_norm.permute(0, 2, 1))
        similarity_av = torch.matmul(emb_a_norm, emb_v_norm.permute(0, 2, 1))
        # Similarity_A = (similarity_at + similarity_av) / 2
        Similarity_A = nn.Softmax(-1)(torch.min(similarity_at, similarity_av))
        ###vision VT, VA
        similarity_vt = torch.matmul(emb_v_norm, emb_t_norm.permute(0, 2, 1))
        similarity_va = torch.matmul(emb_v_norm, emb_a_norm.permute(0, 2, 1))
        # Similarity_V = (similarity_vt + similarity_va) / 2
        Similarity_V = nn.Softmax(-1)(torch.min(similarity_vt, similarity_va))

        return Similarity_T, Similarity_A, Similarity_V
    
    def forward(self, text, audio, vision):
        text, audio, vision = self.text_map(text), self.audio_map(audio), self.vision_map(vision)
        ###map to subspace
        shared_embs, diff_embs = self.map_to_subspace(text, audio, vision)
        return shared_embs, diff_embs
    

class Actor(nn.Module):
    def __init__(self, args):
        super().__init__()
        d_target = args.emb_size
        self.length = args.seqlength
        num_layer = 1
        att_dp = args.atten_dropout
        dp = args.dropout
        n_head = args.n_head

        ####generate action for different representations
        self.text_action = nn.Sequential(nn.Linear(d_target, 1), nn.Sigmoid())
        self.audio_action = nn.Sequential(nn.Linear(d_target, 1), nn.Sigmoid())
        self.vision_action = nn.Sequential(nn.Linear(d_target, 1), nn.Sigmoid())

        layer = nn.TransformerEncoderLayer(d_model=d_target, nhead=n_head, dropout=att_dp)
        self.temporal_t = nn.TransformerEncoder(layer, num_layers=num_layer)
        self.temporal_a = nn.TransformerEncoder(layer, num_layers=num_layer)
        self.temporal_v = nn.TransformerEncoder(layer, num_layers=num_layer)

        self.weight_t = nn.Sequential(nn.Linear(d_target * 2, d_target), nn.Dropout(0), nn.Linear(d_target,1), nn.Sigmoid())
        self.weight_a = nn.Sequential(nn.Linear(d_target * 2, d_target), nn.Dropout(0), nn.Linear(d_target,1), nn.Sigmoid())
        self.weight_v = nn.Sequential(nn.Linear(d_target * 2, d_target), nn.Dropout(0), nn.Linear(d_target,1), nn.Sigmoid())

        ###add
        self.predict = nn.Sequential(     nn.Linear(d_target * 2, d_target),
                                          nn.Dropout(dp),
                                          nn.ReLU(),
                                          nn.BatchNorm1d(d_target),
                                          nn.Linear(d_target, 8))


        ### cat
        # self.predict = nn.Sequential(nn.Linear(d_target * 6, d_target * 3),
        #                                      nn.Dropout(dp),
        #                                      nn.ReLU(),
        #                                      nn.BatchNorm1d(d_target * 3),
        #                                      nn.Linear(d_target * 3, d_target),
        #                                      nn.Dropout(dp),
        #                                      nn.Linear(d_target, 8))


    def forward(self, state):
        state_t, state_a, state_v = state[:, : self.length], state[:, self.length:self.length * 2], state[:, self.length*2:]
        action_t = self.text_action(state_t)
        action_a = self.audio_action(state_a)
        action_v = self.vision_action(state_v)
        action = torch.cat((action_t, action_a, action_v), 1)
        return action
    

    def update_state(self, state, action):
        state_t, state_a, state_v = state[:, : self.length], state[:, self.length:self.length * 2], state[:, self.length*2:]
        action_t, action_a, action_v = action[:, : self.length], action[:, self.length:self.length * 2], action[:, self.length*2:]
        diff_t = torch.mul(state_t, action_t)
        diff_a = torch.mul(state_a, action_a)
        diff_v = torch.mul(state_v, action_v)

        state = torch.cat((diff_t, diff_a, diff_v), 1)
        return state
    
    def predictor(self, shared_embs, diff_embs):
        self.batch = shared_embs.shape[0]
        shared_t, shared_a, shared_v = shared_embs[:, : self.length], shared_embs[:, self.length:self.length * 2], shared_embs[:, self.length*2:]
        diff_t, diff_a, diff_v = diff_embs[:, : self.length], diff_embs[:, self.length:self.length * 2], diff_embs[:, self.length*2:]


        shared_t = shared_t.mean(1)
        shared_a = shared_a.mean(1)
        shared_v = shared_v.mean(1)
        ### temporal features
        diff_t, diff_a, diff_v = diff_t.permute(1, 0, 2), diff_a.permute(1, 0, 2), diff_v.permute(1, 0, 2)
        diff_t, diff_a, diff_v = self.temporal_t(diff_t), self.temporal_a(diff_a), self.temporal_v(diff_v)
        diff_t, diff_a, diff_v = diff_t.permute(1, 0, 2), diff_a.permute(1, 0, 2), diff_v.permute(1, 0, 2)
        diff_t, diff_a, diff_v = diff_t.mean(1).squeeze(), diff_a.mean(1).squeeze(), diff_v.mean(1).squeeze()

        emb_t = torch.cat((shared_t, diff_t), -1)
        emb_a = torch.cat((shared_a, diff_a), -1)
        emb_v = torch.cat((shared_v, diff_v), -1)
        ##add
        weight_t =self.weight_t(emb_t)
        weight_a = self.weight_a(emb_a)
        weight_v = self.weight_v(emb_v)
        weights = torch.cat((weight_t, weight_a, weight_v), -1)
        
        emb_t = torch.mul(emb_t, weights[:, 0].unsqueeze(1))
        emb_a = torch.mul(emb_a, weights[:, 1].unsqueeze(1))
        emb_v = torch.mul(emb_v, weights[:, 2].unsqueeze(1))
        predictions  = self.predict((emb_t + emb_a + emb_v)/3 )
        ##cat
        # predictions = self.predict(torch.cat((emb_t, emb_a, emb_v), -1))
        return predictions.squeeze()

class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        d_target = args.emb_size
        self.length = args.seqlength
        dp = args.dropout
        att_dp = args.atten_dropout
        self.n_head = args.n_head 
        num_layer = 1

        layer = nn.TransformerEncoderLayer(d_model=d_target + self.n_head, nhead=self.n_head, dropout=att_dp)
        self.critic_t = nn.TransformerEncoder(layer, num_layers=num_layer)
        self.critic_a = nn.TransformerEncoder(layer, num_layers=num_layer)
        self.critic_v = nn.TransformerEncoder(layer, num_layers=num_layer)


        self.proj = nn.Sequential(
            nn.Linear((d_target + self.n_head) * 3, d_target * 2),
            nn.Linear(d_target * 2, d_target),
            nn.Dropout(dp),
            nn.ReLU(),
            nn.BatchNorm1d(d_target),
            nn.Linear(d_target, 1))

    def forward(self, state, action):
        b = state.shape[0]
        state_t, state_a, state_v = state[:, :self.length], state[:, self.length:self.length * 2], state[:, self.length * 2:]
        action_t, action_a, action_v = action[:, :self.length], action[:, self.length:self.length * 2], action[:, self.length * 2:]
        
        action_t, action_a, action_v = action_t.repeat(1, 1, self.n_head), action_a.repeat(1, 1, self.n_head), action_v.repeat(1, 1, self.n_head)
        
        text, acoustic, visual = torch.cat((state_t, action_t), -1), torch.cat((state_a, action_a), -1), torch.cat((state_v, action_v), -1)

        text, visual, acoustic = text.permute(1, 0, 2), visual.permute(1, 0, 2), acoustic.permute(1,0,2)
        

        text = self.critic_t(text)
        visual = self.critic_v(visual)
        acoustic = self.critic_a(acoustic)

        text, acoustic, visual = text.permute(1, 0, 2).mean(1), acoustic.permute(1, 0, 2).mean(1), visual.permute(1, 0, 2).mean(1)

        q = self.proj(torch.cat((text, acoustic, visual), -1))
        return q.squeeze()