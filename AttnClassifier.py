import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np
import math
import pdb


class Classifier(nn.Module):
    def __init__(self, n_ways, feat_dim):
        super(Classifier, self).__init__()
        self.calibrator = SupportCalibrator(nway=n_ways, feat_dim=feat_dim, n_head=1)
        self.open_generator = OpenSetGenerater(feat_dim, n_head=1)
        self.metric = Metric_Cosine()
        #self.metric = euclidean_metric()

    def forward(self, features):
        (support_feat, query_feat, openset_feat) = features
        support_feat = torch.mean(support_feat, dim=2)
        supp_protos, support_attn = self.calibrator(support_feat)
        fakeclass_protos, recip_unit = self.open_generator(supp_protos)
        cls_protos = torch.cat([supp_protos, fakeclass_protos], dim=1)
        query_cls_scores = self.metric(cls_protos, query_feat)
        openset_cls_scores = self.metric(cls_protos, openset_feat)
        test_cosine_scores = (query_cls_scores, openset_cls_scores)
        query_funit_distance = 1.0 - self.metric(recip_unit, query_feat)
        qopen_funit_distance = 1.0 - self.metric(recip_unit, openset_feat)
        funit_distance = torch.cat([query_funit_distance, qopen_funit_distance], dim=1)

        return test_cosine_scores, supp_protos, fakeclass_protos, funit_distance


class SupportCalibrator(nn.Module):
    def __init__(self, nway, feat_dim, n_head=1):
        super(SupportCalibrator, self).__init__()
        self.nway = nway
        self.feat_dim = feat_dim
        self.calibrator = MultiHeadAttention(feat_dim // n_head, feat_dim // n_head, (feat_dim, feat_dim))

    def forward(self, support_feat):
        n_bs = support_feat.shape[0]
        support_feat = support_feat.view(-1, 1, self.feat_dim)
        support_center, _, support_attn, _ = self.calibrator(support_feat, support_feat, support_feat, None, None)
        support_center = support_center.view(n_bs, self.nway, -1)
        support_attn = support_attn.view(n_bs, self.nway, -1)
        return support_center, support_attn


class OpenSetGenerater(nn.Module):
    def __init__(self, featdim, n_head=1):
        super(OpenSetGenerater, self).__init__()
        self.att = MultiHeadAttention(featdim // n_head, featdim // n_head, (featdim, featdim))
        self.featdim = featdim
        self.agg_func = nn.Sequential(nn.Linear(featdim, featdim), nn.LeakyReLU(0.5), nn.Dropout(0.5),
                                      nn.Linear(featdim, featdim))

    def forward(self, support_center):
        bs = support_center.shape[0]
        support_center = support_center.view(-1, 1, self.featdim)
        output, attcoef, attn_score, value = self.att(support_center, support_center, support_center, None, None)
        output = output.view(bs, -1, self.featdim)
        fakeclass_center = output.mean(dim=1, keepdim=True)
        fakeclass_center = self.agg_func(fakeclass_center)
        return fakeclass_center, output


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, d_k, d_v, d_model, n_head=1, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        #### Visual feature projection head
        self.w_qs = nn.Linear(d_model[0], n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model[1], n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model[-1], n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model[0] + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model[1] + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model[-1] + d_v)))

        #### Semantic projection head #######
        self.w_qs_sem = nn.Linear(300, n_head * d_k, bias=False)
        self.w_ks_sem = nn.Linear(300, n_head * d_k, bias=False)
        self.w_vs_sem = nn.Linear(300, n_head * d_k, bias=False)

        nn.init.normal_(self.w_qs_sem.weight, mean=0, std=np.sqrt(2.0 / 600))
        nn.init.normal_(self.w_ks_sem.weight, mean=0, std=np.sqrt(2.0 / 600))
        nn.init.normal_(self.w_vs_sem.weight, mean=0, std=np.sqrt(2.0 / 600))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

        self.fc = nn.Linear(n_head * d_v, d_model[0], bias=False)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, q_sem=None, k_sem=None, mark_res=True):
        ### q: bs*nway*D
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        if q_sem is not None:
            sz_b, len_q, _ = q_sem.size()
            sz_b, len_k, _ = k_sem.size()
            q_sem = self.w_qs_sem(q_sem).view(sz_b, len_q, n_head, d_k)
            k_sem = self.w_ks_sem(k_sem).view(sz_b, len_k, n_head, d_k)
            q_sem = q_sem.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
            k_sem = k_sem.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)

        output, attn, attn_score = self.attention(q, k, v, q_sem, k_sem)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        if mark_res:
            output = output + residual

        return output, attn, attn_score, v


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, q_sem=None, k_sem=None):
        attn_score = torch.bmm(q, k.transpose(1, 2))

        if q_sem is not None:
            attn_sem = torch.bmm(q_sem, k_sem.transpose(1, 2))
            q = q + q_sem
            k = k + k_sem
            attn_score = torch.bmm(q, k.transpose(1, 2))

        attn_score /= self.temperature
        attn = self.softmax(attn_score)
        attn = self.dropout(attn)

        output = torch.bmm(attn, v)
        return output, attn, attn_score

class Metric_Cosine(nn.Module):
    def __init__(self, temperature=10):
        super(Metric_Cosine, self).__init__()
        self.temp = nn.Parameter(torch.tensor(float(temperature)))

    def forward(self, supp_center, query_feature):
        ## supp_center: bs*nway*D
        ## query_feature: bs*(nway*nquery)*D
        supp_center = F.normalize(supp_center, dim=-1)  # 1,9,64
        query_feature = F.normalize(query_feature, dim=-1) #1,152,64
        logits = torch.bmm(query_feature, supp_center.transpose(1, 2)) #1,152,9
        return logits * self.temp
