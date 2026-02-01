import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .block import Bottleneck, C2f

__all__ = (
    'CoordAtt','BiFPN_Concat3','BiFPN_Concat2','HSFPN',
    "BiFPN_Concat", "BiFPN", "BiFPN_Transformer", "DynamicBiFPN",
    "FS_Conv", "Hybrid_FS_Conv", "C2f_SCConv",
    "FS_Attention_ECA_CA", "FS_Attention_ECA_SP"
)
#==================================================================================
# from: https://github.com/Changping-Li/YOLOv8_BiFPN.git
class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        return identity * a_h * a_w

class FS_Attention_ECA_SP(nn.Module):
    """超轻量:ECA(通道) + 简化空间门控
    结合了 全局上下文（avg + max pooling） 和 小卷积核，在几乎不增加计算量的前提下显著提升空间注意力能力
    空间注意力应基于 整个特征图的上下文 来判断“哪里重要”，而非局部 3×3 区域。
    全局上下文感知: avg 和 max 池化捕获整个 feature map 的统计信息
    参数极少: 空间分支仅 2 × k × k 个参数（如 k=3 → 18 参数）
    感受野灵活: 可通过 spatial_kernel 调整（默认 3，也可设为 5/7）
    兼容性强: 输出形状与输入完全一致，可无缝插入任何 CNN
    无 bias: 符合注意力模块惯例，减少冗余
    """
    def __init__(self, channels, gamma=2, b=1, spatial_kernel=3):
        super().__init__()
        # ECA: 1D conv on channel avg pool
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.eca_conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()

        # 空间门控：用 depthwise conv 提取空间重要性
        assert spatial_kernel % 2 == 1, "spatial_kernel must be odd"
        self.spatial_conv = nn.Conv2d(
            2, 1, kernel_size=spatial_kernel,
            padding=spatial_kernel // 2, bias=False
        )

    def forward(self, x):
        # ECA
        y = F.adaptive_avg_pool2d(x, 1)               # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)           # (B, 1, C)
        y = self.eca_conv(y)                          # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)         # (B, C, 1, 1)
        x = x * self.sigmoid(y)

        # Spatial Gate
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        s = torch.cat([avg_out, max_out], dim=1)      # (B, 2, H, W)
        s = self.sigmoid(self.spatial_conv(s))        # (B, 1, H, W)
        x = x * s
        return x

class FS_Attention_ECA_CA(nn.Module):
    """超轻量: ECA(通道) + CA(坐标) - 并行融合"""
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        # ECA部分
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.eca_conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        # CA部分
        self.ca = CoordAtt(channels)
        # 可选的融合权重
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        identity = x

        # 计算ECA注意力
        y_eca = F.adaptive_avg_pool2d(x, 1)  # [B, C, 1, 1]
        y_eca = y_eca.squeeze(-1).transpose(-1, -2)  # [B, 1, C]
        y_eca = self.eca_conv(y_eca)  # [B, 1, C]
        y_eca = y_eca.transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        eca_out = self.sigmoid(y_eca)  # 确保在[0,1]

        # 计算CA注意力
        ca_out = self.ca(x)

        # 3. 融合
        out = self.alpha * eca_out + (1 - self.alpha) * ca_out

        return out


class HSFPN(nn.Module):
    def __init__(self, in_planes, ratio = 4, flag=True):
        super(HSFPN, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.flag = flag
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        avg_out = self.conv2(self.relu(self.conv1(self.avg_pool(x))))
        max_out = self.conv2(self.relu(self.conv1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x if self.flag else self.sigmoid(out)

# 结合BiFPN 设置可学习参数 学习不同分支的权重
# 两个分支concat操作
class BiFPN_Concat2(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat2, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1]]
        return torch.cat(x, self.d)


# 三个分支concat操作
class BiFPN_Concat3(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat3, self).__init__()
        self.d = dimension
        # 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并且会向宿主模型注册该参数 成为其一部分 即model.parameters()会包含这个parameter
        # 从而在参数优化的时候可以自动一起优化
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1], weight[2] * x[2]]
        return torch.cat(x, self.d)

#==================================================================================
# from: https://github.com/truongan-2704/yolov8-bifpn-v2
def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class BiFPN_Concat(nn.Module):
    def __init__(self, c1, c2):
        super(BiFPN_Concat, self).__init__()
        self.w1_weight = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2_weight = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = Conv(c1, c2, 1, 1, 0)
        self.act = nn.ReLU()

    def forward(self, x):
        if len(x) == 2:
            w = self.w1_weight
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            x = self.conv(self.act(weight[0] * x[0] + weight[1] * x[1]))
        elif len(x) == 3:
            w = self.w2_weight
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            x = self.conv(self.act(weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2]))
        return x

class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class BiFPN(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(length, dtype=torch.float32), requires_grad=True)
        self.swish = swish()
        self.epsilon = 0.0001

    def forward(self, x):
        weights = self.weight / (torch.sum(self.swish(self.weight), dim=0) + self.epsilon)
        weighted_feature_maps = [weights[i] * x[i] for i in range(len(x))]
        stacked_feature_maps = torch.stack(weighted_feature_maps, dim=0)
        result = torch.sum(stacked_feature_maps, dim=0)
        return result


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mhsa = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)  # (B, Seq_len, Dim)
        attn_output, _ = self.mhsa(x, x, x)
        attn_output = self.norm(attn_output + x)  # Residual Connection
        return attn_output.permute(0, 2, 1).view(b, c, h, w)  # Chuyển về lại


class BiFPN_Transformer(nn.Module):
    def __init__(self, length, embed_dim=128, num_heads=4):
        super().__init__()
        self.length = length
        self.weight = nn.Parameter(torch.ones(length, dtype=torch.float32), requires_grad=True)
        self.epsilon = 1e-4  # Giá trị epsilon nhỏ hơn
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)

    def forward(self, x):
        device = x[0].device  # Lấy thiết bị của tensor đầu vào
        weights = self.weight.to(device)  # Chuyển weight lên đúng thiết bị
        norm_weights = weights / (torch.sum(F.silu(weights), dim=0) + self.epsilon)

        weighted_feature_maps = [norm_weights[i] * x[i] for i in range(self.length)]
        stacked_feature_maps = torch.stack(weighted_feature_maps, dim=0)
        result = torch.sum(stacked_feature_maps, dim=0)

        result = self.attention(result)  # Áp dụng Multi-Head Self-Attention
        return result

#==================================================================================
# from: https://github.com/Gilangarmy/yolov8n-bifpn-v2-.git
"""
BiFPN asli (simplified, faithful implementation) untuk integrasi ke YOLOv8 (Ultralytics).
File ini mendefinisikan:
- SeparableConvBlock: depthwise separable conv + BN + SiLU
- WeightedAdd: fast normalized weighted fusion (ReLU->normalize)
- BiFPNBlock: satu iterasi top-down + bottom-up
- BiFPN: stack beberapa BiFPNBlock (repeats)

Catatan integrasi:
- Masukkan file ini ke: ultralytics/nn/bifpn.py
- Import di tasks.py: from ultralytics.nn.bifpn import BiFPN
- Pastikan fitur masuk (P3,P4,P5) memiliki jumlah channel yang sama (biasanya via 1x1 conv sebelumnya) — saya menambahkan ops untuk menyesuaikan channel jika perlu.
- Di YAML, ganti node-concat/concat2/concat3 yang Anda buat sebelumnya dengan 1 modul BiFPN yang mengambil list fitur input. Contoh penggunaan di parse_model: treat BiFPN as a module that consumes multiple feature maps and outputs same-numbered maps.

Implementasi ini berfokus ke kejelasan dan kompatibilitas dengan pipeline PyTorch/Ultralytics.
"""

class SeparableConvBlock(nn.Module):
    """Depthwise separable conv -> BN -> SiLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class WeightedAdd(nn.Module):
    """Fast normalized fusion used in BiFPN.
    Uses ReLU on weights then normalizes by sum + eps.
    Accepts a list of tensors and returns weighted sum.
    """
    def __init__(self, num_inputs, eps=1e-4):
        super().__init__()
        self.eps = eps
        # initialize with equal importance
        w = torch.ones(num_inputs, dtype=torch.float32)
        self.w = nn.Parameter(w)

    def forward(self, inputs):
        # inputs: list of tensors
        w = F.relu(self.w)
        weight = w / (torch.sum(w) + self.eps)
        out = 0
        for i, t in enumerate(inputs):
            out = out + weight[i] * t
        return out


class BiFPNBlock(nn.Module):
    """
    Single BiFPN block (one top-down pass, one bottom-up pass).
    Expects a list of feature maps ordered from smallest stride (P3) -> larger strides (P4, P5)
    but we'll write to accept inputs as [P3, P4, P5] (P3 highest resolution).
    All features are expected to have the same number of channels. If not, BiFPN will
    adapt by a 1x1 conv to match channels.
    """

    def __init__(self, channels, conv_type=SeparableConvBlock):
        super().__init__()
        C = channels
        # fusion weights
        self.w1 = WeightedAdd(2)  # for top-down merges with 2 inputs
        self.w2 = WeightedAdd(3)  # for bottom-up merges with 3 inputs

        # convs after fusion
        self.p3_td_conv = conv_type(C, C)
        self.p4_td_conv = conv_type(C, C)
        self.p5_td_conv = conv_type(C, C)

        self.p3_bu_conv = conv_type(C, C)
        self.p4_bu_conv = conv_type(C, C)
        self.p5_bu_conv = conv_type(C, C)

        # if needed, adapt channels of inputs
        self.adapt_convs = None

    def adapt_input(self, inputs, channels):
        """Return inputs all adapted to `channels` using 1x1 conv if necessary."""
        adapted = []
        if self.adapt_convs is None:
            self.adapt_convs = nn.ModuleList()
            for t in inputs:
                c = t.shape[1]
                if c != channels:
                    self.adapt_convs.append(nn.Conv2d(c, channels, 1, 1, 0, bias=False))
                else:
                    self.adapt_convs.append(nn.Identity())
        for conv, t in zip(self.adapt_convs, inputs):
            adapted.append(conv(t))
        return adapted

    def forward(self, inputs):
        # inputs: [P3, P4, P5] where P3 has highest spatial resolution
        assert len(inputs) == 3, "BiFPNBlock currently supports exactly 3 levels (P3,P4,P5)"
        # adapt channels if needed
        C = inputs[0].shape[1]
        inputs = self.adapt_input(inputs, C)
        p3, p4, p5 = inputs

        # top-down pathway
        p5_up = F.interpolate(p5, size=(p4.shape[2], p4.shape[3]), mode='nearest')
        p4_td = self.w1([p4, p5_up])
        p4_td = self.p4_td_conv(p4_td)

        p4_up = F.interpolate(p4_td, size=(p3.shape[2], p3.shape[3]), mode='nearest')
        p3_td = self.w1([p3, p4_up])
        p3_td = self.p3_td_conv(p3_td)

        # bottom-up pathway
        p3_down = F.max_pool2d(p3_td, kernel_size=2)
        # combine p3_down, p4, p4_td (three inputs) -> p4_bu
        p4_bu = self.w2([p4, p4_td, p3_down])
        p4_bu = self.p4_bu_conv(p4_bu)

        p4_down = F.max_pool2d(p4_bu, kernel_size=2)
        p5_bu = self.w1([p5, p4_down])
        p5_bu = self.p5_bu_conv(p5_bu)

        return [p3_td, p4_bu, p5_bu]


class ASLI_BiFPN(nn.Module):
    """
    Stack `num_layers` of BiFPNBlock. Input: list of feature maps [P3,P4,P5].
    All outputs keep same channel count as inputs (after optional adapt conv).

    Args:
        channels: number of channels to use internally (if inputs differ, they're adapted)
        num_layers: number of stacked BiFPN blocks (typical: 2 or 3)
    """

    def __init__(self, channels, num_layers=2):
        super().__init__()
        self.channels = channels
        self.num_layers = num_layers
        self.blocks = nn.ModuleList([BiFPNBlock(channels) for _ in range(num_layers)])

    def forward(self, inputs):
        """inputs: list/tuple of 3 tensors [P3,P4,P5]"""
        feats = inputs
        for b in self.blocks:
            feats = b(feats)
        return feats


#=================================================================
# def autopad(k, p=None):  # from YOLOv8
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
#     return p

# class Conv(nn.Module):
#     """Standard convolution (from YOLOv8)"""
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
#         super().__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, dilation=d, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))

class DynamicGateFusion(nn.Module):
    """动态门控融合模块（支持2或3个输入）"""
    def __init__(self, num_inputs, channels):
        super().__init__()
        self.num_inputs = num_inputs
        self.channels = channels

        # 轻量级门控预测器：Conv → SiLU → Conv → Sigmoid
        self.gate_nets = nn.ModuleList([
            nn.Sequential(
                Conv(channels, max(8, channels // 8), k=1, act=True),
                Conv(max(8, channels // 8), channels, k=1, act=False),
                nn.Sigmoid()
            ) for _ in range(num_inputs)
        ])

    def forward(self, inputs):
        # 统一分辨率（以第一个输入为准）
        target_size = inputs[0].shape[2:]
        resized = []
        for x in inputs:
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            resized.append(x)

        # 动态加权
        out = 0
        for i, x in enumerate(resized):
            gate = self.gate_nets[i](x)
            out += gate * x
        return out


class DynamicBiFPNNeck(nn.Module):
    """
    替换 YOLOv8 的 Neck，输入来自 Backbone 的 [P3, P4, P5]
    输出: [P3_out, P4_out, P5_out]（与原 PAN 输出顺序一致）
    """
    def __init__(self, channels=[64, 128, 256]):  # YOLOv8n 的 P3/P4/P5 通道数
        super().__init__()
        c3, c4, c5 = channels, channels, channels #

        # 确保所有层通道一致（YOLOv8 中 P3=64, P4=128, P5=256 → 需统一？）
        # 方案A：不统一通道，分别处理（更高效）
        # 方案B：统一到最大通道（如256）→ 计算量大
        # 这里采用 **方案A：保持原始通道，逐层融合**

        # Top-down path (P5 -> P4 -> P3)
        self.conv_p5 = Conv(c5, c4, k=1)  # 降维到 P4 通道
        self.conv_p4 = Conv(c4, c3, k=1)  # 降维到 P3 通道

        self.fuse_p4_td = DynamicGateFusion(2, c4)
        self.fuse_p3_td = DynamicGateFusion(2, c3)

        # Bottom-up path
        self.down_p3 = Conv(c3, c4, k=3, s=2)  # P3 → P4 尺寸
        self.down_p4 = Conv(c4, c5, k=3, s=2)  # P4 → P5 尺寸

        self.fuse_p4_bu = DynamicGateFusion(3, c4)
        self.fuse_p5_bu = DynamicGateFusion(2, c5)

        # 最终输出层（可选 refine）
        self.out_p3 = Conv(c3, c3, k=3)
        self.out_p4 = Conv(c4, c4, k=3)
        self.out_p5 = Conv(c5, c5, k=3)

    def forward(self, features):
        p3, p4, p5 = features  # 来自 Backbone

        # ---------- Top-down ----------
        p5_up = F.interpolate(p5, size=p4.shape[2:], mode='bilinear', align_corners=False)
        p4_td = self.fuse_p4_td([p4, self.conv_p5(p5_up)])

        p4_up = F.interpolate(p4_td, size=p3.shape[2:], mode='bilinear', align_corners=False)
        p3_td = self.fuse_p3_td([p3, self.conv_p4(p4_up)])

        # ---------- Bottom-up ----------
        p3_down = self.down_p3(p3_td)
        p4_bu = self.fuse_p4_bu([p4_td, p3_down, p4])  # 原始 p4 也参与融合

        p4_down = self.down_p4(p4_bu)
        p5_bu = self.fuse_p5_bu([p5, p4_down])

        # Refine
        out_p3 = self.out_p3(p3_td)
        out_p4 = self.out_p4(p4_bu)
        out_p5 = self.out_p5(p5_bu)

        return [out_p3, out_p4, out_p5]  # 顺序必须与 YOLOv8 Head 期望一致

class DynamicBiFPN(nn.Module):
    """
    Stack `num_layers` of BiFPNBlock. Input: list of feature maps [P3,P4,P5].
    All outputs keep same channel count as inputs (after optional adapt conv).

    Args:
        channels: number of channels to use internally (if inputs differ, they're adapted)
        num_layers: number of stacked BiFPN blocks (typical: 2 or 3)
    """

    def __init__(self, channels, num_layers=2):
        super().__init__()
        self.channels = channels
        self.num_layers = num_layers
        self.blocks = nn.ModuleList([DynamicBiFPNNeck(channels) for _ in range(num_layers)])

    def forward(self, inputs):
        """inputs: list/tuple of 3 tensors [P3,P4,P5]"""
        feats = inputs
        for b in self.blocks:
            feats = b(feats)
        return feats

#========================================================================================================
# fire-smoke conv
class ECA_AttentionGeneral(nn.Module):
    """超轻量：ECA（通道） + 简化空间门控"""
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        # ECA: 1D conv on channel avg pool
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.eca_conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)

        # 空间门控：用 depthwise conv 提取空间重要性
        self.spatial_conv = nn.Conv2d(channels, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # ECA
        y = F.adaptive_avg_pool2d(x, 1).squeeze(-1).transpose(-1, -2)
        y = self.eca_conv(y).transpose(-1, -2).unsqueeze(-1)
        x = x * y.expand_as(x)

        # Spatial Gate
        s = self.sigmoid(self.spatial_conv(x))
        x = x * s
        return x

# class ChannelAttentionGeneral(nn.Module):
#     """通用通道注意力（不依赖输入是否为RGB）"""
#     def __init__(self, in_channels, reduction=16):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels//reduction, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(in_channels//reduction, in_channels, 1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         weight = self.sigmoid(avg_out + max_out)
#         return x * weight

# color_prior 仅用于第一层 RGB
class ChannelAttentionGeneral(nn.Module):
    """针对火焰烟雾颜色特征的通道注意力"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttentionGeneral, self).__init__()
        # 火焰颜色先验（R通道更重要）
        self.color_prior = nn.Parameter(torch.tensor([0.5, 0.3, 0.2]).view(1, 3, 1, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels//reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 增强火焰颜色通道
        if x.shape[1] == 3:  # RGB输入
            x = x * self.color_prior
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

# # 火焰烟雾特征增强卷积
# self.flame_smoke_conv = nn.Sequential(
#             nn.Conv2d(num_channels, num_channels, 3, padding=1, groups=num_channels//16),
#             nn.BatchNorm2d(num_channels),
#             nn.ReLU(),
#             nn.Conv2d(num_channels, num_channels, 1),
#             # 针对火焰的通道注意力
#             ChannelAttention(num_channels, reduction=8)
#         )

#===
class HighFreqConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        # 高频残差连接（类似 Laplacian）
        if k == 3 and s == 1:
            self.use_high_freq = True
            # 固定高频核（不学习）
            hf_kernel = torch.tensor([[-1, -1, -1],
                                      [-1,  8, -1],
                                      [-1, -1, -1]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
            self.register_buffer('hf_kernel', hf_kernel.repeat(c1, 1, 1, 1))
            self.hf_scale = nn.Parameter(torch.tensor(0.2))  # 可学习缩放
        else:
            self.use_high_freq = False

    def forward(self, x):
        out = self.act(self.bn(self.conv(x)))
        if self.use_high_freq:
            # 提取输入高频分量并加到输出（增强边缘/闪烁）
            hf = F.conv2d(x, self.hf_kernel, padding=1, groups=x.shape[1])
            # out = out + 0.2 * hf  # 可学习缩放因子（也可设为可学习参数）
            out = out + self.hf_scale * hf  # 可学习缩放因子（也可设为可学习参数）
        return out

'''
目标	                   推荐方案
追求最高精度(科研/高价值场景)  HybridFlameSmokeConv（输入层 + 关键特征层）
追求实时性（边缘设备）	      FS-Conv（全网络替换）
平衡方案（推荐！）	          Hybrid 用于输入层 + FS-Conv 用于其余层
'''
class FS_Conv(nn.Module):
    """Fire & Smoke Optimized Convolution
        - 仅依赖 高频增强 + 轻量注意力（ECA+空间门控）
        - 无颜色先验，对 RGB 信息利用不足
        - 但 ECA 比 CBAM-style 更适合小通道数

        - ECA 为 1D 卷积，参数极少（≈0）
        - 空间门控用 depthwise conv，轻量
        - 高频核为固定 buffer，无额外参数

    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, use_attention=False):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        # 高频增强（仅 3x3, stride=1）
        self.use_high_freq = (k == 3 and s == 1)
        if self.use_high_freq:
            hf_kernel = torch.tensor([[-1, -1, -1],
                                      [-1,  8, -1],
                                      [-1, -1, -1]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
            self.register_buffer('hf_kernel', hf_kernel.repeat(c1, 1, 1, 1))
            self.hf_scale = nn.Parameter(torch.tensor(0.2))  # 可学习缩放

        # 轻量注意力（按需开启）
        self.attention = ECA_AttentionGeneral(c2) if use_attention else nn.Identity()

    def forward(self, x):
        out = self.act(self.bn(self.conv(x)))

        if self.use_high_freq:
            hf = F.conv2d(x, self.hf_kernel, padding=1, groups=x.shape[1])
            out = out + self.hf_scale * hf

        out = self.attention(out)
        return out

class Hybrid_FS_Conv(nn.Module):
    '''
        - 显式建模 火焰颜色先验（R>G>B），对早期火焰敏感
        - 高频增强 + 通用通道注意力 双重增强
        - 更强的特征判别能力，尤其在复杂背景（如红墙、蒸汽）下

        - CBAM-style 通道注意力：两次 AdaptivePool + FC，FLOPs 较高
        - 若用于多层，累积开销显著
        - 颜色先验仅第一层有效，其余层冗余
    '''
    def __init__(self, c1, c2, k=3, s=1, use_color_prior=False):
        super().__init__()
        # 标准深度可分离卷积（轻量）
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c1, k, s, k//2, groups=c1//16 or 1),
            nn.BatchNorm2d(c1),
            # nn.ReLU(),
            nn.SiLU,
            nn.Conv2d(c1, c2, 1)
        )

        # 高频增强（适用于所有层）
        if k == 3 and s == 1:
            hf_kernel = torch.tensor([[-1, -1, -1],
                                      [-1,  8, -1],
                                      [-1, -1, -1]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
            self.register_buffer('hf_kernel', hf_kernel.repeat(c1,1,1,1))
            self.hf_scale = nn.Parameter(torch.tensor(0.2))
            self.use_hf = True
        else:
            self.use_hf = False

        # 通道注意力（适用于所有层）
        self.ca = ChannelAttentionGeneral(c2, reduction=8)  # 改进版，不依赖RGB

        # 颜色先验（仅当 c1==3 时激活）
        if use_color_prior and c1 == 3:
            self.color_prior = nn.Parameter(torch.tensor([0.5, 0.3, 0.2]).view(1,3,1,1))
        else:
            self.color_prior = None

    def forward(self, x):
        identity = x

        # 颜色先验（仅输入层）
        if self.color_prior is not None:
            x = x * self.color_prior

        # 主卷积
        out = self.conv(x)

        # 高频增强（加在输入上，再进卷积？或加在输出？）
        # 方案：将高频加到主路径输出
        if self.use_hf:
            hf = F.conv2d(identity, self.hf_kernel, padding=1, groups=identity.shape[1])
            hf = F.interpolate(hf, size=out.shape[2:], mode='bilinear') if hf.shape[2:] != out.shape[2:] else hf
            out = out + self.hf_scale * hf

        # 通道注意力
        out = self.ca(out)
        return out

#=====================================================================================
# SCConv（Shifted Convolution）模块。SCConv通过对卷积操作进行优化，提升了对局部特征的学习能力，尤其适用于需要细粒度识别的目标检测任务。
class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int, group_num: int = 16, eps: float = 1e-10):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.randn(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.gamma + self.beta


class SRU(nn.Module):
    def __init__(self, oup_channels: int, group_num: int = 16, gate_threshold: float = 0.5):
        super().__init__()

        self.gn = GroupBatchnorm2d(oup_channels, group_num=group_num)
        self.gate_threshold = gate_threshold
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.gamma / sum(self.gn.gamma)
        reweights = self.sigmoid(gn_x * w_gamma)

        # Gate
        info_mask = reweights >= self.gate_threshold
        noninfo_mask = reweights < self.gate_threshold
        x_1 = info_mask * x
        x_2 = noninfo_mask * x
        x = self.reconstruct(x_1, x_2)
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    '''
    alpha: 0 < alpha < 1
    '''

    def __init__(self, op_channel: int, alpha: float = 1 / 2, squeeze_ratio: int = 2,
                 group_size: int = 2, group_kernel_size: int = 3):
        super().__init__()
        self.up_channel = int(alpha * op_channel)
        self.low_channel = op_channel - self.up_channel
        self.squeeze1 = nn.Conv2d(self.up_channel, self.up_channel // squeeze_ratio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(self.low_channel, self.low_channel // squeeze_ratio, kernel_size=1, bias=False)

        # up
        self.GWC = nn.Conv2d(self.up_channel // squeeze_ratio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(self.up_channel // squeeze_ratio, op_channel, kernel_size=1, bias=False)

        # low
        self.PWC2 = nn.Conv2d(self.low_channel // squeeze_ratio, op_channel - self.low_channel // squeeze_ratio,
                              kernel_size=1, bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)

        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)

        # Fuse
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2

class SCConv(nn.Module):
    # https://github.com/cheng-haha/ScConv/blob/main/ScConv.py
    def __init__(self, op_channel: int, group_num: int = 16, gate_threshold: float = 0.5,
                 alpha: float = 1 / 2, squeeze_ratio: int = 2, group_size: int = 2,
                 group_kernel_size: int = 3):
        super().__init__()
        self.SRU = SRU(op_channel, group_num=group_num, gate_threshold=gate_threshold)
        self.CRU = CRU(op_channel, alpha=alpha, squeeze_ratio=squeeze_ratio,
                       group_size=group_size, group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x

class Bottleneck_SCConv(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = SCConv(c2)

class C2f_SCConv(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_SCConv(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))


# === Minimal test snippet (only runs if file executed directly) ===
if __name__ == '__main__':
    # quick smoke test
    p3 = torch.randn(1, 256, 80, 80)
    p4 = torch.randn(1, 256, 40, 40)
    p5 = torch.randn(1, 256, 20, 20)
    # bifpn = ASLI_BiFPN(256, num_layers=2)
    bifpn = DynamicBiFPN(256, num_layers=2)
    out = bifpn([p3, p4, p5])
    for i, o in enumerate(out):
        print(f'out[{i}]', o.shape)
