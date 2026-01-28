import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())  # type: ignore[no-untyped-call]
print("Number of GPUs:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Current device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
print("Current device properties:", torch.cuda.get_device_properties(torch.cuda.current_device()))
print("cuDNN enabled:", torch.backends.cudnn.enabled)
print("cuDNN deterministic:", torch.backends.cudnn.deterministic)
print("cuDNN benchmark:", torch.backends.cudnn.benchmark)
print("Mem-efficient SDP:", torch.backends.cuda.mem_efficient_sdp_enabled())  # type: ignore[no-untyped-call]

t = torch.tensor([1.0, 2.0, 3.0], device="cuda")
print(t)
