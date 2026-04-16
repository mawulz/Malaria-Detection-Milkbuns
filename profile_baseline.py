import torch
from model_baseline import build_model
import evaluate

def profile():
    model = build_model()
    # Create dummy tensor [batch_size, channels, H, W]
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    assert out.shape == (4, 1), f"Expected shape (4, 1), got {out.shape}"
    print("Baseline Shape test passed!")
    
    print("\nProfiling Baseline Model:")
    evaluate.profile_model(model)

if __name__ == "__main__":
    profile()
