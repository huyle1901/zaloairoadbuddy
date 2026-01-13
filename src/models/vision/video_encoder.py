# src/models/video_encoder.py
import torch
import torch.nn as nn
from typing import Optional, Tuple


from src.models.vision.swin_transformer_v2 import SwinTransformerV2


class SwinV2FrameTokenizer(nn.Module):
    """
    Encode từng FRAME (ảnh) thành spatial tokens bằng Swin V2.

    Input:  x_img: [N, 3, H, W]   (N = B*K*T)
    Output: tokens: [N, L, D]
            - L: số token không gian (ví dụ 49 nếu input 224 và patch merge 4 stage)
            - D: dim embedding cuối (ví dụ 768 với cấu hình mặc định)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer=nn.LayerNorm,
        ape: bool = False,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        pretrained_window_sizes=(0, 0, 0, 0),
    ):
        super().__init__()

        # num_classes=0 để bỏ head classification (head=Identity)
        self.backbone = SwinTransformerV2(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=0,
            embed_dim=embed_dim,
            depths=list(depths),
            num_heads=list(num_heads),
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            pretrained_window_sizes=list(pretrained_window_sizes),
        )

    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        """
        Return token sequence, NOT pooled vector.

        x_img: [N, 3, 224, 224]
        tokens: [N, L, D]
        """
        # ======== copy logic from SwinTransformerV2.forward_features BUT stop before avgpool ========
        x = self.backbone.patch_embed(x_img)  # [N, L0, C0] e.g. [N, 3136, 96]
        if self.backbone.ape:
            x = x + self.backbone.absolute_pos_embed
        x = self.backbone.pos_drop(x)

        for layer in self.backbone.layers:
            x = layer(x)  # cuối: [N, L, D] e.g. [N, 49, 768]
        
        x = self.backbone.norm(x)  # [N, L, D]
        return x


class VideoEncoderSwinV2Tokens(nn.Module):
    """
    Video encoder cho input từ dataloader của bạn:

    Input:
        video: [B, K, T, C, H, W]  (C=3, H=W=224 thường)
    Output:
        video_tokens: [B, S, D_out]
            - S = K*T*L (L là số spatial tokens/frame, vd 49)
        video_attn_mask: [B, S]  (1 = token thật, 0 = padding) -> ở đây luôn 1 hết

    """

    def __init__(
        self,
        img_size: int = 224,
        swin_embed_dim: int = 96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size: int = 7,
        out_dim: Optional[int] = None,   # nếu muốn project về dim khác (vd dim của text)
        token_pool: Optional[str] = None # None = giữ spatial tokens; "mean" = pool L -> 1 token/frame
    ):
        super().__init__()

        self.frame_encoder = SwinV2FrameTokenizer(
            img_size=img_size,
            embed_dim=swin_embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
        )

        # dim cuối của Swin = embed_dim * 2^(num_layers-1)
        self.swin_out_dim = self.frame_encoder.backbone.num_features  # vd 768

        self.token_pool = token_pool

        if out_dim is None:
            self.out_dim = self.swin_out_dim
            self.proj = nn.Identity()
        else:
            self.out_dim = out_dim
            self.proj = nn.Linear(self.swin_out_dim, out_dim)

    def forward(
        self,
        video: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        video: [B,K,T,C,H,W]

        return:
          video_tokens: [B, S, D_out]
          video_attn_mask: [B, S] (ones)
        """
        B, K, T, C, H, W = video.shape

        # 1) gộp batch để chạy Swin cho từng frame
        x = video.reshape(B * K * T, C, H, W)  # [B*K*T, C, H, W]

        # 2) Swin trả spatial tokens cho từng frame
        tokens = self.frame_encoder(x)  # [B*K*T, L, D]

        # 3) (tuỳ chọn) pool spatial tokens L -> 1 token/frame
        if self.token_pool == "mean":
            tokens = tokens.mean(dim=1, keepdim=True)  # [B*K*T, 1, D]

        # 4) project dim nếu cần (để match text dim)
        tokens = self.proj(tokens)  # [B*K*T, L_or_1, D_out]

        # 5) reshape về [B, K, T, L, D_out]
        L = tokens.shape[1]
        tokens = tokens.view(B, K, T, L, self.out_dim)

        # 6) flatten thành sequence token toàn video: S = K*T*L
        video_tokens = tokens.reshape(B, K * T * L, self.out_dim)  # [B, S, D_out]

        # 7) mask: vì S cố định (K,T,L fixed) => toàn 1
        video_attn_mask = torch.ones((B, K * T * L), dtype=torch.long, device=video.device)

        return video_tokens, video_attn_mask
