import torch
from model import build_model
import evaluate

def test_dimensions():
    model = build_model()
    # Create dummy tensor [batch_size, channels, H, W]
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    # For BCEWithLogits, expected shape is [batch_size, 1]
    assert out.shape == (4, 1), f"Expected shape (4, 1), got {out.shape}"
    print("Shape test passed!")
    
    print("\nProfiling Model:")
    evaluate.profile_model(model)

if __name__ == "__main__":
    test_dimensions()
