import torch, ctypes
print(torch.__version__)       # 2.7.1
print(torch.version.cuda)      # 12.6
print(torch.cuda.is_available())
ctypes.CDLL("libcublasLt.so.12")  # should load without error
