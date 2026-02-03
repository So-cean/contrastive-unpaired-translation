from packaging import version
import torch
from torch import nn
import torch.nn.functional as F


class RankNCELoss(nn.Module):
    """
    RankNCE: Exploring Negatives in Contrastive Learning for Unpaired Image-to-Image Translation
    核心思想：不再使用所有非局部 patch 作为负样本，而是选择高质量的负样本
    通过基于互信息贡献的排序，排除可能的假阴性（False Negatives）
    """
    def __init__(self, opt, top_k_ratio=0.5, bottom_k_ratio=0.1):
        """
        Args:
            opt: 配置选项
            top_k_ratio: 选择最难负样本的比例（高相似度，难区分）
            bottom_k_ratio: 排除过于简单负样本的比例（低相似度，无信息）
        """
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        
        # RankNCE 特定参数
        self.top_k_ratio = getattr(opt, 'ranknce_top_k', top_k_ratio)
        self.bottom_k_ratio = getattr(opt, 'ranknce_bottom_k', bottom_k_ratio)
        
    def forward(self, feat_q, feat_k):
        """
        Args:
            feat_q: query features [num_patches, dim] 或 [batch, num_patches, dim]
            feat_k: key features [num_patches, dim] 或 [batch, num_patches, dim]
        """
        batch_dim = feat_q.shape[0]
        
        # 处理维度
        if feat_q.dim() == 2:
            # 单张图片情况：[num_patches, dim]
            num_patches = feat_q.shape[0]
            dim = feat_q.shape[1]
            feat_k = feat_k.detach()
            
            # pos logit: [num_patches, 1]
            l_pos = torch.bmm(
                feat_q.view(num_patches, 1, -1), 
                feat_k.view(num_patches, -1, 1)
            ).view(num_patches, 1)
            
            # 完整的相似度矩阵（用于排序选择负样本）
            full_sim_matrix = torch.mm(feat_q, feat_k.t())  # [num_patches, num_patches]
            
            # 选择高质量负样本（RankNCE 核心）
            l_neg = self._select_ranked_negatives(full_sim_matrix, num_patches)
            
        else:
            # batch 情况
            if self.opt.nce_includes_all_negatives_from_minibatch:
                batch_dim_for_bmm = 1
                feat_q = feat_q.view(1, -1, feat_q.size(-1))
                feat_k = feat_k.view(1, -1, feat_k.size(-1))
            else:
                batch_dim_for_bmm = self.opt.batch_size
                feat_q = feat_q.view(batch_dim_for_bmm, -1, feat_q.size(-1))
                feat_k = feat_k.view(batch_dim_for_bmm, -1, feat_k.size(-1))
            
            feat_k = feat_k.detach()
            npatches = feat_q.size(1)
            
            # pos logit: [batch, npatches, 1]
            l_pos = torch.bmm(
                feat_q.view(batch_dim_for_bmm, npatches, 1, -1),
                feat_k.view(batch_dim_for_bmm, npatches, -1, 1)
            ).view(batch_dim_for_bmm * npatches, 1)
            
            # 完整的相似度矩阵
            full_sim_matrix = torch.bmm(feat_q, feat_k.transpose(2, 1))  # [batch, npatches, npatches]
            full_sim_matrix = full_sim_matrix.view(-1, npatches)  # [batch*npatches, npatches]
            
            # 选择高质量负样本
            l_neg = self._select_ranked_negatives_batch(full_sim_matrix, npatches, batch_dim_for_bmm)
        
        # 合并正负样本
        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T
        
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))
        
        return loss
    
    def _select_ranked_negatives(self, sim_matrix, num_patches):
        """
        单张图片情况：基于排序选择高质量负样本
        
        策略：
        1. 排除对角线（正样本）
        2. 排除相似度过高的（可能是假阴性）
        3. 排除相似度过低的（无信息）
        4. 保留中间的"高质量"负样本
        """
        device = sim_matrix.device
        
        # 排除对角线（正样本位置）
        mask = torch.eye(num_patches, device=device, dtype=torch.bool)
        sim_masked = sim_matrix.masked_fill(mask, float('-inf'))
        
        # 计算每个 query 应该选择多少个负样本
        num_negatives = num_patches - 1  # 排除正样本
        k_top = max(1, int(num_negatives * self.top_k_ratio))
        k_bottom = max(0, int(num_negatives * self.bottom_k_ratio))
        k_select = k_top - k_bottom  # 实际选择的数量
        
        # 对每个 query，选择排名在 [k_bottom, k_top] 之间的负样本
        # 相似度排序：从高到低
        sorted_sim, sorted_indices = torch.sort(sim_masked, dim=1, descending=True)
        
        # 创建选择掩码
        # 保留排序在 [k_bottom, k_top) 之间的样本
        ranks = torch.arange(num_patches, device=device).unsqueeze(0).expand(num_patches, -1)
        select_mask = (ranks >= k_bottom) & (ranks < k_top) & (sorted_sim > float('-inf') / 2)
        
        # 构建最终的负样本 logits
        # 未选择的设为很小的值（exp(-10) ≈ 0）
        l_neg = torch.full_like(sim_masked, -10.0)
        l_neg.scatter_(1, sorted_indices, sorted_sim.masked_fill(~select_mask, -10.0))
        
        return l_neg
    
    def _select_ranked_negatives_batch(self, sim_matrix, npatches, batch_size):
        """
        batch 情况下的负样本选择
        sim_matrix: [batch*npatches, npatches]
        """
        device = sim_matrix.device
        total_queries = sim_matrix.size(0)
        
        # 构建 block 对角掩码（排除同一张图片内的正样本）
        # sim_matrix 已经展平，需要排除每 npatches 个中的对角线
        mask = torch.zeros(total_queries, npatches, device=device, dtype=torch.bool)
        for b in range(batch_size):
            start = b * npatches
            end = start + npatches
            mask[start:end, :] = torch.eye(npatches, device=device, dtype=torch.bool)
        
        sim_masked = sim_matrix.masked_fill(mask, float('-inf'))
        
        # 选择参数
        num_negatives = npatches
        k_top = max(1, int(num_negatives * self.top_k_ratio))
        k_bottom = max(0, int(num_negatives * self.bottom_k_ratio))
        
        # 排序选择
        sorted_sim, sorted_indices = torch.sort(sim_masked, dim=1, descending=True)
        ranks = torch.arange(npatches, device=device).unsqueeze(0).expand(total_queries, -1)
        select_mask = (ranks >= k_bottom) & (ranks < k_top) & (sorted_sim > float('-inf') / 2)
        
        l_neg = torch.full_like(sim_masked, -10.0)
        l_neg.scatter_(1, sorted_indices, sorted_sim.masked_fill(~select_mask, -10.0))
        
        return l_neg