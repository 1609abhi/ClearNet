import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


##########################################################################
## Tensor reshaping helpers
##########################################################################
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


##########################################################################
## CoordAtt + DEM  (Degradation Estimation Module, Sec 3.1)
##########################################################################
class h_sigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = h_sigmoid()
    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1  = nn.Conv2d(inp, mip, 1)
        self.act    = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, 1)
        self.conv_w = nn.Conv2d(mip, oup, 1)

    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y   = self.act(self.conv1(torch.cat([x_h, x_w], dim=2)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        return self.conv_h(x_h).sigmoid() * self.conv_w(x_w).sigmoid()  # (B, C, H, W) in (0,1)


class DEM(nn.Module):
    """
    Degradation Estimation Module (Sec 3.1) — produces:
      deg_map     : (B, n_feat, H, W)  spatial degradation map in (0,1)
      restored_img: (B, 3,      H, W)  auxiliary supervised output
    """
    def __init__(self, n_feat):
        super().__init__()
        self.coord_att = CoordAtt(n_feat, n_feat)
        self.conv1     = nn.Conv2d(n_feat, 3,      kernel_size=1)
        self.conv2     = nn.Conv2d(3,      n_feat, kernel_size=1)

    def forward(self, x, x_img):
        # x     : (B, n_feat, H, W)  feature map at this level
        # x_img : (B, 3,      H, W)  RGB image at same spatial resolution
        residual     = self.conv1(x)               # (B, 3,      H, W) correction
        restored_img = residual + x_img             # (B, 3,      H, W) aux output
        mag          = self.conv2(residual).abs()   # (B, n_feat, H, W)
        deg_map      = self.coord_att(mag)          # (B, n_feat, H, W) in (0,1)
        return deg_map, restored_img


##########################################################################
## Task-Conditional RMSNorm  (DAN core, Eq. 3-5)
##########################################################################
class RMSNorm(nn.Module):
    """
    Spatial + task-conditional RMSNorm operating on (B, H*W, C).

    weight = 1 + spatial_gate(deg_map * task_embed)   reshaped to (B, H*W, C)
    out    = (x / RMS(x)) * weight
    """
    def __init__(self, dim, num_tasks, eps=1e-5):
        super().__init__()
        self.eps          = eps
        self.task_bank    = nn.Parameter(torch.randn(num_tasks, dim))
        # depthwise conv learns to route spatial degradation info into the scale
        self.spatial_gate = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        nn.init.zeros_(self.spatial_gate.weight)
        nn.init.zeros_(self.spatial_gate.bias)

    def forward(self, x, logits, deg_map):
        # x      : (B, H*W, C)
        # logits : (B, num_tasks)
        # deg_map: (B, C,   H, W)
        task_embed = (logits @ self.task_bank)[:, :, None, None]   # (B, C, 1, 1)
        combined   = deg_map * task_embed                            # (B, C, H, W)
        weight_4d  = self.spatial_gate(combined)                     # (B, C, H, W)
        weight_3d  = rearrange(weight_4d, 'b c h w -> b (h w) c')   # (B, H*W, C)
        weight     = 1 + weight_3d                                   # zero-init → identity at start

        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * weight


class DAN(nn.Module):
    """Degradation-Aware Normalization (Sec 3.2) — 4D wrapper around RMSNorm."""
    def __init__(self, dim, num_tasks, LayerNorm_type=None):
        super().__init__()
        self.body = RMSNorm(dim, num_tasks)

    def forward(self, x, logits, deg_map):
        # x: (B, C, H, W)
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x), logits, deg_map), h, w)


##########################################################################
## Gated-DConv Feed-Forward Network (GDFN, Sec 3.3)
##########################################################################
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden           = int(dim * ffn_expansion_factor)
        self.project_in  = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv      = nn.Conv2d(hidden * 2, hidden * 2, 3, 1, 1,
                                     groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


##########################################################################
## Multi-DConv Head Transposed Attention (MDTA, Sec 3.3)
##########################################################################
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads   = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv         = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv  = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1,
                                      groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        out  = attn.softmax(dim=-1) @ v
        out  = rearrange(out, 'b head c (h w) -> b (head c) h w',
                         head=self.num_heads, h=h, w=w)
        return self.project_out(out)


##########################################################################
## TransformerBlock (Eq. 6-7) — DAN replaces standard LayerNorm
##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias,
                 num_tasks, LayerNorm_type=None):
        super().__init__()
        self.norm1 = DAN(dim, num_tasks, LayerNorm_type)
        self.attn  = Attention(dim, num_heads, bias)
        self.norm2 = DAN(dim, num_tasks, LayerNorm_type)
        self.ffn   = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, logits, deg_map):
        x = x + self.attn(self.norm1(x, logits, deg_map))
        x = x + self.ffn(self.norm2(x, logits, deg_map))
        return x


##########################################################################
## TransformerSequential  — threads logits + deg_map through all blocks
##########################################################################
class TransformerSequential(nn.Module):
    def __init__(self, *blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, logits, deg_map):
        for block in self.blocks:
            x = block(x, logits, deg_map)
        return x


##########################################################################
## Patch embed / Downsample / Upsample  — unchanged
##########################################################################
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, 3, 1, 1, bias=bias)
    def forward(self, x):
        return self.proj(x)

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1, bias=False),
            nn.PixelUnshuffle(2)
        )
    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, 3, 1, 1, bias=False),
            nn.PixelShuffle(2)
        )
    def forward(self, x):
        return self.body(x)


##########################################################################
## Restormer with DEM-based spatial degradation maps (ClearNet)
##########################################################################
class Restormer(nn.Module):
    """
    ClearNet: Restormer backbone with:
      - DEM at every encoder/decoder stage (Sec 3.1)
      - DAN replacing LayerNorm in every TransformerBlock (Sec 3.2)
      - Pyramid of auxiliary outputs for multi-scale supervision (Sec 3.4)

    forward() returns a list:
      [aux_enc1, aux_enc2, aux_enc3, aux_mid,
       aux_dec3, aux_dec2, aux_dec1, aux_ref,
       final_output]

    For inference / eval, use model(inp, logits)[-1].
    For training, supervise all outputs with build_gt_pyramid().
    """
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',   # kept for API compat, ignored
                 dual_pixel_task=False,
                 num_tasks=5):
        super().__init__()

        def make_block(d, h):
            return TransformerBlock(dim=d, num_heads=h,
                                    ffn_expansion_factor=ffn_expansion_factor,
                                    bias=bias, num_tasks=num_tasks)

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # ── Encoder ──────────────────────────────────────────────────────────
        self.dem_enc1       = DEM(dim)
        self.encoder_level1 = TransformerSequential(
            *[make_block(dim, heads[0]) for _ in range(num_blocks[0])])
        self.down1_2        = Downsample(dim)

        self.dem_enc2       = DEM(dim * 2)
        self.encoder_level2 = TransformerSequential(
            *[make_block(dim * 2, heads[1]) for _ in range(num_blocks[1])])
        self.down2_3        = Downsample(dim * 2)

        self.dem_enc3       = DEM(dim * 4)
        self.encoder_level3 = TransformerSequential(
            *[make_block(dim * 4, heads[2]) for _ in range(num_blocks[2])])
        self.down3_4        = Downsample(dim * 4)

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.dem_mid        = DEM(dim * 8)
        self.latent         = TransformerSequential(
            *[make_block(dim * 8, heads[3]) for _ in range(num_blocks[3])])

        # ── Decoder ──────────────────────────────────────────────────────────
        self.up4_3              = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=bias)
        self.dem_dec3           = DEM(dim * 4)
        self.decoder_level3     = TransformerSequential(
            *[make_block(dim * 4, heads[2]) for _ in range(num_blocks[2])])

        self.up3_2              = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=bias)
        self.dem_dec2           = DEM(dim * 2)
        self.decoder_level2     = TransformerSequential(
            *[make_block(dim * 2, heads[1]) for _ in range(num_blocks[1])])

        self.up2_1              = Upsample(dim * 2)
        # after cat with enc skip: dim + dim = dim*2
        self.dem_dec1           = DEM(dim * 2)
        self.decoder_level1     = TransformerSequential(
            *[make_block(dim * 2, heads[0]) for _ in range(num_blocks[0])])

        self.dem_ref            = DEM(dim * 2)
        self.refinement         = TransformerSequential(
            *[make_block(dim * 2, heads[0]) for _ in range(num_refinement_blocks)])

        self.dual_pixel_task = dual_pixel_task
        if dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, dim * 2, 1, bias=bias)

        self.output = nn.Conv2d(dim * 2, out_channels, 3, 1, 1, bias=bias)

    # -------------------------------------------------------------------------
    def forward(self, inp_img, logits):
        # Downsampled RGB references for DEM at each scale
        img_s1 = inp_img                                                              # (B, 3, H,   W  )
        img_s2 = F.interpolate(inp_img, scale_factor=0.5,   mode='bilinear', align_corners=False)
        img_s4 = F.interpolate(inp_img, scale_factor=0.25,  mode='bilinear', align_corners=False)
        img_s8 = F.interpolate(inp_img, scale_factor=0.125, mode='bilinear', align_corners=False)

        # ── Encoder ──────────────────────────────────────────────────────────
        inp_enc1             = self.patch_embed(inp_img)
        deg_map1, aux_enc1   = self.dem_enc1(inp_enc1, img_s1)
        out_enc1             = self.encoder_level1(inp_enc1, logits, deg_map1)

        inp_enc2             = self.down1_2(out_enc1)
        deg_map2, aux_enc2   = self.dem_enc2(inp_enc2, img_s2)
        out_enc2             = self.encoder_level2(inp_enc2, logits, deg_map2)

        inp_enc3             = self.down2_3(out_enc2)
        deg_map3, aux_enc3   = self.dem_enc3(inp_enc3, img_s4)
        out_enc3             = self.encoder_level3(inp_enc3, logits, deg_map3)

        # ── Bottleneck ───────────────────────────────────────────────────────
        inp_lat              = self.down3_4(out_enc3)
        deg_map_mid, aux_mid = self.dem_mid(inp_lat, img_s8)
        latent               = self.latent(inp_lat, logits, deg_map_mid)

        # ── Decoder ──────────────────────────────────────────────────────────
        inp_dec3             = self.reduce_chan_level3(
                                   torch.cat([self.up4_3(latent), out_enc3], dim=1))
        deg_map_d3, aux_dec3 = self.dem_dec3(inp_dec3, img_s4)
        out_dec3             = self.decoder_level3(inp_dec3, logits, deg_map_d3)

        inp_dec2             = self.reduce_chan_level2(
                                   torch.cat([self.up3_2(out_dec3), out_enc2], dim=1))
        deg_map_d2, aux_dec2 = self.dem_dec2(inp_dec2, img_s2)
        out_dec2             = self.decoder_level2(inp_dec2, logits, deg_map_d2)

        inp_dec1             = torch.cat([self.up2_1(out_dec2), out_enc1], dim=1)
        deg_map_d1, aux_dec1 = self.dem_dec1(inp_dec1, img_s1)
        out_dec1             = self.decoder_level1(inp_dec1, logits, deg_map_d1)

        deg_map_ref, aux_ref = self.dem_ref(out_dec1, img_s1)
        out_ref              = self.refinement(out_dec1, logits, deg_map_ref)

        # ── Final output ─────────────────────────────────────────────────────
        if self.dual_pixel_task:
            final = self.output(out_ref + self.skip_conv(inp_enc1))
        else:
            final = self.output(out_ref) + inp_img

        # enc (fine→coarse) | bottleneck | dec (coarse→fine) | refinement | final
        return [aux_enc1, aux_enc2, aux_enc3, aux_mid,
                aux_dec3, aux_dec2, aux_dec1, aux_ref,
                final]


##########################################################################
## Pyramid GT builder  — use in training loop
##########################################################################
def build_gt_pyramid(gt, outputs):
    """
    Returns a list of GT tensors matching each output's spatial resolution.

    Usage in train loop:
        outputs = model(inp, logits)
        gt_list = build_gt_pyramid(gt, outputs)
        loss    = sum(criterion(pred, target)
                      for pred, target in zip(outputs, gt_list))
    """
    gt_list = []
    for out in outputs:
        if out.shape[-2:] == gt.shape[-2:]:
            gt_list.append(gt)
        else:
            gt_list.append(F.interpolate(gt, size=out.shape[-2:],
                                         mode='bilinear', align_corners=False))
    return gt_list


##########################################################################
## Sanity test
##########################################################################
def test_restormer():
    print("Testing ClearNet (Restormer + DEM + DAN)")
    B, C, H, W = 2, 3, 256, 256
    num_tasks  = 5
    device     = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    x      = torch.randn(B, C, H, W, device=device, requires_grad=True)
    logits = F.one_hot(torch.randint(0, num_tasks, (B,)),
                       num_classes=num_tasks).float().to(device)

    model = Restormer(
        inp_channels=C, out_channels=C, dim=48,
        num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
        heads=[1, 2, 4, 8], num_tasks=num_tasks
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    outputs = model(x, logits)
    print(f"  Number of outputs   : {len(outputs)}")
    for i, o in enumerate(outputs):
        print(f"    output[{i}]: {tuple(o.shape)}")

    assert outputs[-1].shape == x.shape, "Final output shape mismatch"

    gt_list = build_gt_pyramid(x, outputs)
    loss = sum(F.l1_loss(p, t) for p, t in zip(outputs, gt_list))
    loss.backward()
    assert x.grad is not None, "Gradients did not flow to input"
    print("  Forward + pyramid loss + backward: OK")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_restormer()
    print("\nAll tests passed.")
