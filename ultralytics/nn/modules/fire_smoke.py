import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = (
    'CoordAtt','BiFPN_Concat3','BiFPN_Concat2','HSFPN',
    "BiFPN_Concat", "BiFPN", "BiFPN_Transformer",

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

# === Minimal test snippet (only runs if file executed directly) ===
if __name__ == '__main__':
    # quick smoke test
    p3 = torch.randn(1, 256, 80, 80)
    p4 = torch.randn(1, 256, 40, 40)
    p5 = torch.randn(1, 256, 20, 20)
    bifpn = ASLI_BiFPN(256, num_layers=2)
    out = bifpn([p3, p4, p5])
    for i, o in enumerate(out):
        print(f'out[{i}]', o.shape)
