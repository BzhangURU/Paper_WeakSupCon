
from __future__ import print_function

import torch
import torch.nn as nn

class SimilarityLoss(nn.Module):
    """features_neg is usually set to None"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SimilarityLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features_pos, features_neg=None):
        if features_neg is None:
            features=features_pos
            n_pos = features_pos.size(0)
            labels = torch.ones(n_pos, dtype=torch.int64)
            mask = None
        else:

            features = torch.cat([features_pos, features_neg], dim=0)
            
            n_pos = features_pos.size(0)
            n_neg = features_neg.size(0)

            # Create the tensor with ones for n_pos and zeros for n_neg
            labels = torch.cat([torch.ones(n_pos, dtype=torch.int64), torch.zeros(n_neg, dtype=torch.int64)])
            mask = None

            WeakSupCon_pos_samples_selection=torch.cat([torch.ones(n_pos, dtype=torch.bool), 
                                                        torch.zeros(n_neg, dtype=torch.bool), 
                                                        torch.ones(n_pos, dtype=torch.bool), 
                                                        torch.zeros(n_neg, dtype=torch.bool)])

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)#Computes element-wise equality
        else:
            mask = mask.float().to(device)#dimension [batch_size, batch_size]

        contrast_count = features.shape[1]#n_views
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]#features: [bsz, n_views, ...], so only get n_views==0
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)#matrix multiplication
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        
        logits = anchor_dot_contrast - logits_max.detach()
        
        mask = mask.repeat(anchor_count, contrast_count)#anchor_count==contrast_count==n_views==2, repeat(n_views,n_views)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,#the dimension along which to scatter
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),#index tensor, specifying which elements to modify
            0
        )
        
        mask = mask * logits_mask
        
        if features_neg is None:
            log_prob = logits
            mask_pos_pairs = mask.sum(1)
            mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)# avoid division by 0
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs #dimension [batch_size * n_views]
        else:
            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask#dimension [batch_size * n_views, batch_size * n_views]
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
            log_prob_selected_rows=log_prob[WeakSupCon_pos_samples_selection]
            mask_selected_rows=mask[WeakSupCon_pos_samples_selection]

            mask_pos_pairs = mask_selected_rows.sum(1)
            mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)# avoid division by 0
            mean_log_prob_pos = (mask_selected_rows * log_prob_selected_rows).sum(1) / mask_pos_pairs #dimension [batch_size * n_views]

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, n_pos).mean()
        return loss
