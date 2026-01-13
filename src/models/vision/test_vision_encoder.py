import torch
from src.models.vision.video_encoder import VideoEncoderSwinV2Tokens

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, K, T, C, H, W = 2, 8, 8, 3, 224, 224
    video = torch.randn(B, K, T, C, H, W, device=device)

    enc = VideoEncoderSwinV2Tokens(
        img_size=224,
        swin_embed_dim=96,
        depths=(2,2,6,2),
        num_heads=(3,6,12,24),
        window_size=7,
        out_dim=None,          # giữ nguyên 768
        token_pool=None        # GIỮ 49 token / frame
    ).to(device)

    enc.eval()
    with torch.no_grad():
        video_tokens, video_attn_mask = enc(video)

    print("video_tokens:", video_tokens.shape)   # [B, S, D]
    print("video_attn_mask:", video_attn_mask.shape)

    # S = K*T*L, với img 224 + Swin mặc định => L=49
    expected_L = 49
    expected_S = K * T * expected_L
    expected_D = enc.out_dim

    assert video_tokens.shape == (B, expected_S, expected_D), "Shape video_tokens sai"
    assert video_attn_mask.shape == (B, expected_S), "Shape mask sai"
    assert (video_attn_mask == 1).all(), "Mask phải toàn 1 vì K,T,L cố định"

    print("Unit test OK")

if __name__ == "__main__":
    main()
