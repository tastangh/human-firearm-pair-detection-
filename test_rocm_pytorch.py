import torch

def test_rocm():
    if torch.backends.mps.is_available():
        print("MPS backend is available. (This is for Apple M1/M2, not ROCm)")
    if torch.version.hip:
        print(f"ROCm version detected: {torch.version.hip}")
    else:
        print("ROCm not detected by PyTorch.")

    print("Checking for available devices:")
    if torch.cuda.is_available():
        print("torch.cuda.is_available(): True")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Device capability: {torch.cuda.get_device_capability(0)}")
        x = torch.randn(3, 3).to("cuda")
        print("Tensor moved to ROCm GPU:\n", x)
    else:
        print("No ROCm-compatible GPU found by torch.cuda.")

if __name__ == "__main__":
    test_rocm()
