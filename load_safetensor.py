
# %%
import torch
from rich.console import Console
import safetensors


# %%
print(torch.__version__)
print(torch.cuda.is_available())
# %%



# open safetensors file

data = safetensors.safe_open("epiNoiseoffset_v2.safetensors", framework='pt')

# %%
