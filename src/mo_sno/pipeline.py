import os
print(os.environ.get("CUDA_VISIBLE_DEVICES"))

import numba.cuda
print(numba.cuda.is_available())
print(numba.cuda.list_devices())