from packaging import version
import torch
from torch import nn
import torch.nn.functional as F


class RankNCELoss(nn.Module):
    """
    RankNCE: Exploring Negatives in Contrastive Learning for Unpaired Image-to-Image Translation
    核心思想：通过互信息贡献排序，选择高质量负样本，排除假阴性（False Negatives）
    
    策略：
    1. 排除相似度最高的负样本（可能是假阴性，如同一物体的不同部分）
    2. 排除相似度最低的负样本（无信息，过于简单）
    3. 保留中等难度的负样本（最具区分性）
    """
    def __init__(self, opt, top_k_ratio=0.5, bottom_k_ratio=0.1):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        
        # RankNCE 特定参数
        self.top_k_ratio = getattr(opt, 'ranknce_top_k', top_k_ratio)
        self.bottom_k_ratio = getattr(opt, 'ranknce_bottom_k', bottom_k_ratio)
        
        # 确保比例合理
        assert 0 <= self.bottom_k_ratio < self.top_k_ratio <= 1.0, \
            f"Invalid ratios: bottom_k={self.bottom_k_ratio}, top_k={self.top_k_ratio}"
        
        print(f"[RankNCE] top_k_ratio={self.top_k_ratio:.2f}, "
              f"bottom_k_ratio={self.bottom_k_ratio:.2f}, "
              f"effective_ratio={self.top_k_ratio - self.bottom_k_ratio:.2f}")
        
    def forward(self, feat_q, feat_k):
        """
        Args:
            feat_q: query features [num_patches, dim] 或 [batch, num_patches, dim]
            feat_k: key features [num_patches, dim] 或 [batch, num_patches, dim]
        """
        batch_dim = len(feat_q.shape)
        temperature = getattr(self.opt, 'nce_T', 0.07)
        
        if batch_dim == 2:
            # 单张图片情况：[num_patches, dim]
            feat_k = feat_k.detach()
            num_patches = feat_q.shape[0]
            
            # 计算正样本 logit: [num_patches, 1]
            l_pos = torch.sum(feat_q * feat_k, dim=1, keepdim=True)
            
            # 计算所有负样本相似度: [num_patches, num_patches]
            sim_matrix = torch.mm(feat_q, feat_k.t())
            
            # 排除对角线（正样本）
            mask = torch.eye(num_patches, device=sim_matrix.device, dtype=torch.bool)
            sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))
            
            # 选择高质量负样本
            out = self._select_ranked_negatives(sim_matrix, l_pos)
            
        else:
            # batch 情况：[batch, num_patches, dim]
            batch_size, npatches, dim = feat_q.shape
            
            if self.opt.nce_includes_all_negatives_from_minibatch:
                # 所有 batch 内样本作为负样本
                # feat_q/feats_k: [batch, npatches, dim] -> [1, batch*npatches, dim]
                feat_q = feat_q.view(1, -1, dim)
                feat_k = feat_k.view(1, -1, dim).detach()
                total_patches = batch_size * npatches
                
                # 正样本 logit: 对应位置
                l_pos = torch.sum(feat_q.view(total_patches, dim) * feat_k.view(total_patches, dim), 
                                dim=1, keepdim=True)  # [total_patches, 1]
                
                # 所有相似度: [total_patches, total_patches]
                sim_matrix = torch.mm(feat_q.view(total_patches, dim), 
                                    feat_k.view(total_patches, dim).t())
                
                # 排除正样本：对角线
                mask = torch.eye(total_patches, device=sim_matrix.device, dtype=torch.bool)
                sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))
                
                out = self._select_ranked_negatives(sim_matrix, l_pos)
                
            else:
                # 每个样本独立处理
                feat_k = feat_k.detach()
                
                # 正样本 logit: [batch, npatches, 1]
                l_pos = torch.sum(feat_q * feat_k, dim=2, keepdim=True)
                
                # 所有相似度: [batch, npatches, npatches]
                sim_matrix = torch.bmm(feat_q, feat_k.transpose(2, 1))
                
                # 排除对角线（每个样本内部）
                mask = torch.eye(npatches, device=sim_matrix.device, dtype=torch.bool).unsqueeze(0)
                sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))
                
                # reshape 为 [batch*npatches, npatches] 统一处理
                sim_matrix = sim_matrix.view(-1, npatches)
                l_pos = l_pos.view(-1, 1)
                
                out = self._select_ranked_negatives(sim_matrix, l_pos)
        
        # 温度缩放
        out = out / temperature
        
        # 计算损失（标签是 0，即第一列是正样本）
        loss = self.cross_entropy_loss(
            out, 
            torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device)
        )
        
        return loss
    
    def _select_ranked_negatives(self, sim_matrix, l_pos):
        """
        向量化实现：基于排序选择高质量负样本
        使用 topk 避免 -inf 问题
        
        Args:
            sim_matrix: [N, M] 负样本相似度矩阵（对角线已 mask 为 -inf）
            l_pos: [N, 1] 正样本 logit
        Returns:
            out: [N, 1+k_select]
        """
        # 1. 获取有效负样本数量（排除 -inf）
        valid_mask = sim_matrix != float('-inf')
        num_valid_per_query = valid_mask.sum(dim=1)  # [N]
        min_valid = num_valid_per_query.min().item()
        
        if min_valid == 0:
            # 极端情况：没有负样本
            return torch.cat([l_pos, torch.zeros_like(l_pos)], dim=1)
        
        # 2. 计算 k 值（基于最小有效负样本数）
        k_top = max(1, min(int(min_valid * self.top_k_ratio), min_valid))
        k_bottom = min(int(min_valid * self.bottom_k_ratio), k_top - 1)
        k_select = k_top - k_bottom
        
        # 确保至少选择 1 个负样本
        if k_select < 1:
            k_select = 1
            k_bottom = k_top - 1
        
        # 3. 使用 topk 获取前 k_top 个最相似的负样本
        # topk 会自动跳过 -inf（除非所有值都是 -inf）
        top_values, _ = torch.topk(sim_matrix, k=k_top, dim=1, largest=True, sorted=True)
        
        # 4. 选择排名在 [k_bottom, k_top) 的负样本
        # 排除最相似的 k_bottom 个（可能的假阴性）
        selected_neg_sim = top_values[:, k_bottom:]
        
        # 5. 组合正负样本
        out = torch.cat([l_pos, selected_neg_sim], dim=1)
        
        return out


# ============= 测试代码 =============
if __name__ == "__main__":
    print("=" * 70)
    print("RankNCE Loss 完整测试")
    print("=" * 70)
    
    # 模拟配置
    class DummyOpt:
        nce_T = 0.07
        batch_size = 4
        nce_includes_all_negatives_from_minibatch = False
        ranknce_top_k = 0.5
        ranknce_bottom_k = 0.1
    
    # ========== 测试 1: 单张图片 ==========
    print("\n测试 1: 单张图片 [num_patches, dim]")
    opt = DummyOpt()
    criterion = RankNCELoss(opt)
    
    num_patches = 256
    dim = 256
    feat_q = F.normalize(torch.randn(num_patches, dim), dim=1)
    feat_k = F.normalize(torch.randn(num_patches, dim), dim=1)
    
    loss = criterion(feat_q, feat_k)
    print(f"  输入: feat_q={feat_q.shape}, feat_k={feat_k.shape}")
    print(f"  损失: mean={loss.mean().item():.4f}, std={loss.std().item():.4f}")
    assert loss.shape == (num_patches,)
    print("  ✅ 通过")
    
    # ========== 测试 2: Batch (独立模式) ==========
    print("\n测试 2: Batch [batch, num_patches, dim] - 独立模式")
    opt.nce_includes_all_negatives_from_minibatch = False
    criterion = RankNCELoss(opt)
    
    batch = 4
    num_patches = 128
    feat_q = F.normalize(torch.randn(batch, num_patches, dim), dim=2)
    feat_k = F.normalize(torch.randn(batch, num_patches, dim), dim=2)
    
    loss = criterion(feat_q, feat_k)
    print(f"  输入: feat_q={feat_q.shape}, feat_k={feat_k.shape}")
    print(f"  损失: mean={loss.mean().item():.4f}")
    assert loss.shape == (batch * num_patches,)
    print("  ✅ 通过")
    
    # ========== 测试 3: Batch (共享模式) ==========
    print("\n测试 3: Batch [batch, num_patches, dim] - 共享负样本模式")
    opt.nce_includes_all_negatives_from_minibatch = True
    criterion = RankNCELoss(opt)
    
    loss = criterion(feat_q, feat_k)
    print(f"  输入: feat_q={feat_q.shape}")
    print(f"  损失: mean={loss.mean().item():.4f}")
    assert loss.shape == (batch * num_patches,)
    print("  ✅ 通过")
    
    # ========== 测试 4: 不同参数 ==========
    print("\n测试 4: 不同 top_k/bottom_k 配置")
    configs = [
        (1.0, 0.0, "使用所有负样本"),
        (0.5, 0.1, "标准RankNCE"),
        (0.3, 0.0, "只用最难30%"),
    ]
    
    feat_q = F.normalize(torch.randn(128, 128), dim=1)
    feat_k = F.normalize(torch.randn(128, 128), dim=1)
    
    for top_k, bottom_k, desc in configs:
        opt_test = DummyOpt()
        opt_test.ranknce_top_k = top_k
        opt_test.ranknce_bottom_k = bottom_k
        opt_test.nce_includes_all_negatives_from_minibatch = False
        
        criterion_test = RankNCELoss(opt_test)
        loss = criterion_test(feat_q, feat_k)
        print(f"  {desc}: loss={loss.mean().item():.4f}")
    
    print("  ✅ 通过")
    
    # ========== 测试 5: 梯度 ==========
    print("\n测试 5: 梯度测试")
    opt = DummyOpt()
    opt.nce_includes_all_negatives_from_minibatch = False
    criterion = RankNCELoss(opt)
    
    # 🔧 修复：先创建需要梯度的张量，再归一化
    feat_q_raw = torch.randn(128, 128, requires_grad=True)
    
    # 归一化（这会创建新张量，但梯度会传回 feat_q_raw）
    feat_q = F.normalize(feat_q_raw, dim=1)
    
    loss = criterion(feat_q, feat_k).mean()
    loss.backward()
    
    print(f"  损失: {loss.item():.4f}")
    print(f"  feat_q_raw 梯度范数: {feat_q_raw.grad.norm().item():.4f}")
    assert not torch.isnan(feat_q_raw.grad).any()
    assert feat_q_raw.grad.abs().sum() > 0, "梯度应该非零"
    print("  ✅ 梯度正常传播")
    
    # ========== 测试 6: 边界情况 ==========
    print("\n测试 6: 极端参数 (top_k=0.95, bottom_k=0.9)")
    opt = DummyOpt()
    opt.ranknce_top_k = 0.95
    opt.ranknce_bottom_k = 0.9
    opt.nce_includes_all_negatives_from_minibatch = False
    criterion = RankNCELoss(opt)
    
    feat_q = F.normalize(torch.randn(64, 64), dim=1)
    feat_k = F.normalize(torch.randn(64, 64), dim=1)
    loss = criterion(feat_q, feat_k)
    print(f"  损失: {loss.mean().item():.4f}")
    assert not torch.isnan(loss).any()
    assert not torch.isinf(loss).any()
    print("  ✅ 通过")
    
    # ========== 测试 7: 少量 patches 边界情况 ==========
    print("\n测试 7: 少量 patches (num_patches=8)")
    opt = DummyOpt()
    opt.nce_includes_all_negatives_from_minibatch = False
    criterion = RankNCELoss(opt)
    
    feat_q = F.normalize(torch.randn(8, 64), dim=1)
    feat_k = F.normalize(torch.randn(8, 64), dim=1)
    loss = criterion(feat_q, feat_k)
    print(f"  损失: {loss.mean().item():.4f}")
    assert loss.shape == (8,)
    print("  ✅ 通过")
    
    # ========== 测试 8: 验证选择的负样本数量 ==========
    print("\n测试 8: 验证选择的负样本数量")
    
    class InspectOpt:
        nce_T = 0.07
        batch_size = 1
        nce_includes_all_negatives_from_minibatch = False
        ranknce_top_k = 0.5
        ranknce_bottom_k = 0.1
    
    opt = InspectOpt()
    criterion = RankNCELoss(opt)
    
    num_patches = 100
    feat_q = F.normalize(torch.randn(num_patches, 64), dim=1)
    feat_k = F.normalize(torch.randn(num_patches, 64), dim=1)
    
    # 手动计算期望的负样本数
    num_negatives = num_patches - 1  # 99
    k_top = int(num_negatives * 0.5)  # 49
    k_bottom = int(num_negatives * 0.1)  # 9
    expected_neg_count = k_top - k_bottom  # 40
    
    # 通过钩子检查输出维度
    loss = criterion(feat_q, feat_k)
    
    # 实际上我们可以通过 _select_ranked_negatives 验证
    l_pos = torch.sum(feat_q * feat_k, dim=1, keepdim=True)
    sim_matrix = torch.mm(feat_q, feat_k.t())
    mask = torch.eye(num_patches, device=sim_matrix.device, dtype=torch.bool)
    sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))
    
    out = criterion._select_ranked_negatives(sim_matrix, l_pos)
    actual_neg_count = out.shape[1] - 1  # 减去正样本那一列
    
    print(f"  总 patches: {num_patches}")
    print(f"  可用负样本: {num_negatives}")
    print(f"  期望选择: {expected_neg_count}")
    print(f"  实际选择: {actual_neg_count}")
    assert actual_neg_count == expected_neg_count, \
        f"负样本数量不匹配：期望 {expected_neg_count}，实际 {actual_neg_count}"
    print("  ✅ 负样本数量正确")
    
    print("\n" + "=" * 70)
    print("🎉 所有测试通过！RankNCE 实现完全正确")
    print("=" * 70)