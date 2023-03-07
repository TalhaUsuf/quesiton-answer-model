# %%
import torch
from rich.console import Console
import torchvision


# %%
# load fashion mnist dataset

trf = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=3),
    torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.5,), (0.5,))])

dataset = torchvision.datasets.FashionMNIST(download=True, root='data', transform=trf)

# %%
# import random_split
from torch.utils.data import random_split

total_sz = len(dataset)

test_size = int(0.3 * total_sz)


# %%


train, test = random_split(dataset, [total_sz - test_size, test_size])


# %%
from torch.utils.data import DataLoader


# %%

train_dl = DataLoader(train, batch_size=32, shuffle=True, num_workers=4)
test_dl = DataLoader(test, batch_size=32, shuffle=True, num_workers=4)


# %%
# load resnet model
# model = torchvision.models.resnet18(pretrained=True)
# %%
import torch_tensorrt
# %%


# calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
#     test_dl,
#     cache_file="./calibration.cache",
#     use_cache=False,
#     algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION,
#     device=torch.device("cuda:0"),
# )


# # %%
# trt_mod = torch_tensorrt.compile(model, inputs=[torch_tensorrt.Input((1, 3, 28, 28))],
#                                     enabled_precisions={torch.float, torch.half, torch.int8},
#                                     calibrator=calibrator,
#                                     device={
#                                          "device_type": torch_tensorrt.DeviceType.GPU,
#                                          "gpu_id": 0,
#                                          "dla_core": 0,
#                                          "allow_gpu_fallback": False,
#                                          "disable_tf32": False
#                                      })
# %%



from pytorch_quantization import quant_modules
import pytorch_quantization.nn as quant_nn
from pytorch_quantization.tensor_quant import QuantDescriptor
quant_modules.initialize()
# %%
quant_desc_input = QuantDescriptor(calib_method='histogram')
quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

model = torchvision.models.resnet18(pretrained=True)
model.cuda()
# %%
model.eval()
# %%

# calibrate


from tqdm.auto import tqdm

def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistic"""

    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
        model(image.cuda())
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()
                
                 
# 

# %%
from pytorch_quantization import calib
def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(F"{name:40}: {module}")
    model.cuda()

# It is a bit slow since we collect histograms on CPU
with torch.no_grad():
    collect_stats(model, test_dl, num_batches=10)
    compute_amax(model, method="percentile", percentile=99.99)
# %%





# %%

model.to('cpu')
inp = torch.randn(1, 3, 28, 28)
torch.onnx.export(model,  inp ,"resnet18_quant.onnx" ,input_names=['input'], output_names=['output'], do_constant_folding=True, opset_version=14,
                  dynamic_axes={
                      'input' : {
                          0:'batch',
                          2:'height',
                          3:'width'
                      }
                  })
# %%
