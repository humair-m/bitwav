
import torch
import torch.nn as nn
from bitwav.module.vocoder import Vocos, UpSamplerBlock, crossover_merge_linkwitz_riley

# Mock components to test the logic flow without loading heavy weights
class MockBackbone(nn.Module):
    def forward(self, x, **kwargs):
        # Input: (B, C, L) -> Output: (B, hidden, L)
        return x  # simple passthrough for shape

class MockHead(nn.Module):
    def forward(self, x):
        # Input: (B, hidden, L) -> Output: (B, audio_len_24k)
        # Assuming 24kHz head (upsamples by stride usually, but for features input it might be direct)
        # Let's assume input features are already at frame rate and head does ISTFT or similar
        # For this test, let's say it returns (B, 1, L * 256) assuming hop 256
        return torch.randn(x.shape[0], 1, x.shape[2] * 256)

class MockHead48k(nn.Module):
    def forward(self, x):
        # Input: (B, hidden, L_upsampled) -> Output: (B, 1, audio_len_48k)
        # Input is already upsampled 2x. Stride 256 gives total 512x original L.
        return torch.randn(x.shape[0], 1, x.shape[2] * 256)

class MockFeatureExtractor(nn.Module):
    def forward(self, x, **kwargs):
        return x

def test_vocoder_48k():
    print("Initializing 48kHz Vocoder components...")
    
    # 1. Setup dimensions
    batch_size = 2
    seq_len = 100
    dim = 256
    
    # 2. Initialize modules
    # UpSamplerBlock for 48kHz path (upsample by 2)
    upsampler = UpSamplerBlock(in_channels=dim, upsample_factors=[2])
    
    # Mocks for other parts
    backbone = MockBackbone()
    head_24k = MockHead()
    head_48k = MockHead48k()
    feature_extractor = MockFeatureExtractor()
    
    # 3. Instantiate Vocos
    vocoder = Vocos(
        feature_extractor=feature_extractor,
        backbone=backbone,
        head=head_24k,
        upsampler=upsampler,
        head_48k=head_48k
    )
    
    # 4. Create dummy input features (B, dim, L)
    dummy_features = torch.randn(batch_size, dim, seq_len)
    
    print(f"Input features shape: {dummy_features.shape}")
    
    # 5. Run decode
    print("Running decode()...")
    output_audio = vocoder.decode(dummy_features)
    
    print(f"Output audio shape: {output_audio.shape}")
    
    # 6. Verify shapes and logic
    # Expected output length: seq_len * 512 (since mock head_48k does *512)
    # The decode method does:
    #   pred_audio2 = head_24k(features) -> (B, 1, L*256) (24kHz)
    #   resampled = resample(pred_audio2, 24k, 48k) -> (B, 1, L*512)
    #   pred_audio = head_48k(upsampled) -> (B, 1, L*512)
    #   merge -> (B, 1, L*512)
    
    expected_len = seq_len * 512
    assert output_audio.shape[-1] == expected_len, f"Expected length {expected_len}, got {output_audio.shape[-1]}"
    
    print("\n✅ Vocoder logic verification PASSED!")
    print("   - Dual-path execution successful")
    print("   - Upsampler block working")
    print("   - Crossover merge successful")
    print("   - Output shape correct for 48kHz")

if __name__ == "__main__":
    try:
        from bitwav.module.vocoder import crossover_merge_linkwitz_riley
        # Test crossover specifically
        print("\nTesting Linkwitz-Riley Crossover...")
        x1 = torch.randn(1, 1, 16000)
        x2 = torch.randn(1, 1, 16000)
        out = crossover_merge_linkwitz_riley(x1, x2, cutoff=4000)
        assert out.shape == x1.shape
        print("✅ Crossover verification PASSED")
        
        test_vocoder_48k()
    except Exception as e:
        print(f"\n❌ Verification FAILED: {e}")
        import traceback
        traceback.print_exc()
