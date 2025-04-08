import torch
import argparse
import os
import torch.backends.cudnn as cudnn
from architecture import *
from utils_hsi import save_matv73
import cv2
import numpy as np
import itertools

# ✅ Disable GPU usage
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ✅ Setup parser with hardcoded values
parser = argparse.ArgumentParser(description="SSR")
args = parser.parse_args([])  # Empty args to skip CLI
args.method = 'mst_plus_plus'
args.pretrained_model_path = './model_zoo/mst_plus_plus.pth'
args.rgb_path = './demo/ARAD_1K_0912.jpg'
args.outf = './exp/mst_plus_plus/'
args.ensemble_mode = 'mean'
args.gpu_id = 'cpu'

# ✅ Use these args as your config
opt = args
device = torch.device('cpu')  # Force CPU

# ✅ Create output folder if it doesn't exist
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

def main():
    cudnn.benchmark = False  # Disable GPU-specific benchmarking
    pretrained_model_path = opt.pretrained_model_path
    method = opt.method
    
    # ✅ Load model on CPU
    model = model_generator(method, pretrained_model_path)
    model = model.to(device)
    model.eval()

    test(model, opt.rgb_path, opt.outf)

def test(model, rgb_path, save_path):
    var_name = 'cube'
    bgr = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = np.float32(rgb)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    rgb = np.expand_dims(np.transpose(rgb, [2, 0, 1]), axis=0).copy()
    
    # ✅ Send tensor to CPU
    rgb = torch.from_numpy(rgb).float().to(device)

    print(f'Reconstructing {rgb_path}')
    with torch.no_grad():
        result = forward_ensemble(rgb, model, opt.ensemble_mode)

    result = result.cpu().numpy() * 1.0
    result = np.transpose(np.squeeze(result), [1, 2, 0])
    result = np.minimum(result, 1.0)
    result = np.maximum(result, 0)

    mat_name = rgb_path.split('/')[-1][:-4] + '.mat'
    mat_dir = os.path.join(save_path, mat_name)
    save_matv73(mat_dir, var_name, result)
    print(f'The reconstructed hyper spectral image is saved as {mat_dir}.')

def forward_ensemble(x, forward_func, ensemble_mode='mean'):
    def _transform(data, xflip, yflip, transpose, reverse=False):
        if not reverse:  # forward transform
            if xflip:
                data = torch.flip(data, [3])
            if yflip:
                data = torch.flip(data, [2])
            if transpose:
                data = torch.transpose(data, 2, 3)
        else:  # reverse transform
            if transpose:
                data = torch.transpose(data, 2, 3)
            if yflip:
                data = torch.flip(data, [2])
            if xflip:
                data = torch.flip(data, [3])
        return data

    outputs = []
    opts = itertools.product((False, True), (False, True), (False, True))
    for xflip, yflip, transpose in opts:
        data = x.clone()
        data = _transform(data, xflip, yflip, transpose)
        data = forward_func(data)
        outputs.append(_transform(data, xflip, yflip, transpose, reverse=True))

    if ensemble_mode == 'mean':
        return torch.stack(outputs, 0).mean(0)
    elif ensemble_mode == 'median':
        return torch.stack(outputs, 0).median(0)[0]

if __name__ == '__main__':
    main()
