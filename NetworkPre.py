import torch.nn as nn
import torch
import torch.nn.functional as F
from AttnClassifier import Classifier
from BACKBONE import Backbone


class FeatureNet(nn.Module):
    def __init__(self, n_ways, feature_dim, data_info):
        super(FeatureNet, self).__init__()
        self.feat_dim = feature_dim
        self.n_ways = n_ways
        self.info = data_info
        # self.feature = Backbone(self.feat_dim)
        self.feature = Backbone(data_info)
        self.cls_classifier = Classifier(self.n_ways, self.feat_dim)
        # self.feature.apply(weights_init)

    def forward(self, the_input, labels=None, test=False):
        the_sizes = [_.size(1) for _ in the_input]
        (ne, _, nc, nh, nw) = the_input[0].size()
        combined_data = torch.cat(the_input, dim=1).view(-1, nc, nh, nw)
        combined_feat = self.feature(combined_data)
        support_feat, query_feat, supopen_feat, openset_feat = torch.split(combined_feat.view(ne, -1, self.feat_dim),
                                                                           the_sizes, dim=1)
        (support_label, query_label, supopen_label, openset_label) = labels
        cls_label = torch.cat([query_label, openset_label], dim=1)

        ### First Task
        support_feat = support_feat.view(ne, self.n_ways, -1, self.feat_dim)
        if test:
            test_cosine_scores = self.task_proto(
                (support_feat, query_feat, openset_feat), cls_label, test)
            test_cls_probs = self.task_pred(test_cosine_scores[0], test_cosine_scores[1])
            return test_cls_probs[0]
        else:
            test_cosine_scores, supp_protos, fakeclass_protos, loss_cls, loss_funit = self.task_proto(
                (support_feat, query_feat, openset_feat), cls_label, test)
            test_cls_probs = self.task_pred(test_cosine_scores[0], test_cosine_scores[1])

        ## Second task
        supopen_feat = supopen_feat.view(ne, self.n_ways, -1, self.feat_dim)
        _, supp_protos_aug, fakeclass_protos_aug, loss_cls_aug, loss_funit_aug = self.task_proto(
            (supopen_feat, openset_feat, query_feat), cls_label, test)

        supp_protos = F.normalize(supp_protos, dim=-1)
        fakeclass_protos = F.normalize(fakeclass_protos, dim=-1)
        supp_protos_aug = F.normalize(supp_protos_aug, dim=-1)
        fakeclass_protos_aug = F.normalize(fakeclass_protos_aug, dim=-1)

        loss_open_hinge = 0.0
        # loss_open_hinge_1 = F.mse_loss(fakeclass_protos.repeat(1,self.n_ways, 1), supp_protos)
        # loss_open_hinge_2 = F.mse_loss(fakeclass_protos_aug.repeat(1,self.n_ways, 1), supp_protos_aug)
        # loss_open_hinge = loss_open_hinge_1 + loss_open_hinge_2

        loss = (loss_cls + loss_cls_aug, loss_open_hinge, loss_funit + loss_funit_aug)
        return test_cls_probs, loss

    def task_proto(self, features, cls_label, test):
        if test:
            test_cosine_scores, supp_protos, fakeclass_protos, funit_distance = self.cls_classifier(features)
            return test_cosine_scores
        else:      
            test_cosine_scores, supp_protos, fakeclass_protos, funit_distance = self.cls_classifier(features)
            (query_cls_scores, openset_cls_scores) = test_cosine_scores
            cls_scores = torch.cat([query_cls_scores, openset_cls_scores], dim=1)
            fakeunit_loss = fakeunit_compare(funit_distance, cls_label, self.n_ways)
            cls_scores, close_label, cls_label = cls_scores.view(-1, self.n_ways + 1), cls_label[:,
                                                                                    :query_cls_scores.size(1)].reshape(
                -1), cls_label.view(-1)
            loss_cls = F.cross_entropy(cls_scores, cls_label)
            return test_cosine_scores, supp_protos, fakeclass_protos, loss_cls, fakeunit_loss

    def task_pred(self, query_cls_scores, openset_cls_scores):
        query_cls_probs = F.softmax(query_cls_scores.detach(), dim=-1)
        openset_cls_probs = F.softmax(openset_cls_scores.detach(), dim=-1)
        return (query_cls_probs, openset_cls_probs)


class Metric_Cosine(nn.Module):
    def __init__(self, temperature=10):
        super(Metric_Cosine, self).__init__()
        self.temp = nn.Parameter(torch.tensor(float(temperature)))

    def forward(self, supp_center, query_feature):
        ## supp_center: bs*nway*D
        ## query_feature: bs*(nway*nquery)*D
        supp_center = F.normalize(supp_center, dim=-1)  # eps=1e-6 default 1e-12
        query_feature = F.normalize(query_feature, dim=-1)
        logits = torch.bmm(query_feature, supp_center.transpose(1, 2))
        return logits * self


def fakeunit_compare(funit_distance, cls_label, n_ways):
    cls_label_binary = F.one_hot(cls_label, n_ways + 1)[:, :, :-1].float()
    loss = torch.sum(F.binary_cross_entropy_with_logits(input=funit_distance, target=cls_label_binary))
    return loss
