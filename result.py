import os

import SimpleITK
import SimpleITK as sitk
from options.test_options import TestOptions
from utils.NiftiDataset import *
import utils.NiftiDataset as NiftiDataset_testing
from torch.utils.data import DataLoader
from models import create_model

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics import PeakSignalNoiseRatio
import time
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn

from scipy import linalg
# ===========================================================
from torch.nn.functional import adaptive_avg_pool2d
from torch.nn.functional import mse_loss
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 載入預訓練的3D影像分類模型
model_inception_v3 = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
from scipy import ndimage


torch_ssim = StructuralSimilarityIndexMeasure(data_range=255)
torch_psnr = PeakSignalNoiseRatio(data_range=255)

def calculate_background(image):
    background = ndimage.gaussian_filter(image, sigma=50)
    background_mean = np.mean(background)
    background_std = np.std(background)

    return background_mean, background_std

def calculate_signal(image, background_mean):
    signal = image - background_mean
    signal_mean = np.mean(signal)
    return signal_mean

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    """計算FID的Frechet距離"""
    sigma1 = torch.from_numpy(sigma1.reshape(1, 1))

    sigma2 = torch.from_numpy(sigma2.reshape(1, 1))
    covmean = torch.sqrt(sigma1 @ sigma2).unsqueeze(0).double().cuda()
    if torch.any(torch.isnan(covmean)):
        return float("nan")
    mean_diff = (mu1 - mu2).double().cuda().reshape(1, 1)
    return (mean_diff @ mean_diff + torch.trace(sigma1) + torch.trace(sigma2) - 2 * torch.trace(covmean.reshape(1, 1))).item()

def calculate_activation_statistics(images, model, batch_size=1):
    """計算激活統計信息"""
    model.eval().cuda()
    act = []
    # pred = torch.zeros([images.shape[2], 1000])
    with torch.no_grad():
        for start_idx in range(0, images.shape[2]):
            batch_var = images[:, :, start_idx, :, :].cuda()

            if batch_var.shape[1] == 1:
                batch_var = batch_var.repeat(1, 3, 1, 1)
            if batch_var.shape[1] != 3:
                raise ValueError('Input images should be either grayscale or RGB')
            y = model(batch_var)
            pred = y[0]
            pred = torch.reshape(pred, (pred.shape[0], 1, 1))
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1)).squeeze(-1).squeeze(-1)
            act.append(pred.cpu().detach().numpy())
    act = torch.from_numpy(np.concatenate(act, axis=0)).double()
    mu = torch.mean(act, dim=0)
    sigma = np.cov(act.numpy(), rowvar=False)
    return mu, sigma

def calculate_fid(images1, images2, model, batch_size):
    """計算FID"""
    model = model.eval()
    mu1, sigma1 = calculate_activation_statistics(images1, model, batch_size)
    mu2, sigma2 = calculate_activation_statistics(images2, model, batch_size)
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid


# ===========================================================

def from_numpy_to_itk(image_np, image_itk):
    image_np = np.transpose(image_np, (2, 1, 0))
    image = sitk.GetImageFromArray(image_np)
    image.SetOrigin(image_itk.GetOrigin())
    image.SetDirection(image_itk.GetDirection())
    image.SetSpacing(image_itk.GetSpacing())
    return image




def inference(image_path, label_path, filename, txt_path, result_path, i):

    # read image file
    reader = sitk.ImageFileReader()
    reader.SetFileName(image_path)
    image = reader.Execute()

    reader.SetFileName(label_path)
    real = reader.Execute()

    castImageFilter = sitk.CastImageFilter()
    input_pixel_type = image.GetPixelID()
    castImageFilter.SetOutputPixelType(input_pixel_type)
    # castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
    image = castImageFilter.Execute(image)
    real = castImageFilter.Execute(real)

    image = Normalization(image)
    real = Normalization(real)

    pred = sitk.GetArrayFromImage(image)
    target = sitk.GetArrayFromImage(real)

    writer = sitk.ImageFileWriter()

    target = np.transpose(target, (2, 1, 0))
    label = from_numpy_to_itk(target, real)
    result = os.path.join(result_path, str(i) + "_label.nii")
    writer.SetFileName(result)
    writer.Execute(label)

    pred = np.transpose(pred, (2, 1, 0))
    image = from_numpy_to_itk(pred, image)
    result = os.path.join(result_path, str(i) + "_image.nii")
    writer.SetFileName(result)
    writer.Execute(image)

    image_pred = np.transpose(pred, (2, 1, 0))
    image_target = np.transpose(target, (2, 1, 0))

    # start_time = time.time()
    t_target = torch.from_numpy(target).unsqueeze(0)
    t_pred = torch.from_numpy(pred).unsqueeze(0)
    ssim_result = torch_ssim(t_target, t_pred)
    psnr_result = torch_psnr(t_target, t_pred)

    ssim_result = ssim_result.numpy()
    psnr_result = psnr_result.numpy()

    global PSNR
    global SSIM
    global FID


    PSNR = psnr(target, pred, data_range=255)
    SSIM = ssim(target, pred, data_range=255)

    # 計算FID分數
    pred = np.repeat(np.repeat(np.repeat(pred, 2.5, axis=0), 2, axis=1), 2, axis=2)
    target = np.repeat(np.repeat(np.repeat(target, 2.5, axis=0), 2, axis=1), 2, axis=2)
    pred = np.transpose(pred, (2, 1, 0))
    target = np.transpose(target, (2, 1, 0))
    fid_pred = np.expand_dims(pred.astype(np.float32), axis=(0, 1))
    fid_target = np.expand_dims(target.astype(np.float32), axis=(0, 1))
    fid_pred = torch.from_numpy(fid_pred)
    fid_target = torch.from_numpy(fid_target)

    FID = calculate_fid(fid_target, fid_pred, model_inception_v3, batch_size=1)

    # FID = 0

    fake_image_sp = np.transpose(fake_image_sp, (2, 1, 0))
    fake_image_nsp = np.transpose(fake_image_nsp, (2, 1, 0))
    real_image_sp = np.transpose(real_image_sp, (2, 1, 0))
    real_image_nsp = np.transpose(real_image_nsp, (2, 1, 0))

    fake_image_sp = from_numpy_to_itk(fake_image_sp, image)
    fake_image_nsp = from_numpy_to_itk(fake_image_nsp, image)
    real_image_sp = from_numpy_to_itk(real_image_sp, real)
    real_image_nsp = from_numpy_to_itk(real_image_nsp, real)
    writer = sitk.ImageFileWriter()


    result = os.path.join(result_path, str(i) + "_fake_image_sp.nii")
    writer.SetFileName(result)
    writer.Execute(fake_image_sp)

    result = os.path.join(result_path, str(i) + "_fake_image_nsp.nii")
    writer.SetFileName(result)
    writer.Execute(fake_image_nsp)

    result = os.path.join(result_path, str(i) + "_real_image_sp.nii")
    writer.SetFileName(result)
    writer.Execute(real_image_sp)

    result = os.path.join(result_path, str(i) + "_real_image_nsp.nii")
    writer.SetFileName(result)
    writer.Execute(real_image_nsp)




    with open(txt_path, "a") as f:
        f.write("=======================\n")
        f.write(filename)
        f.write("\nPSNR: {:.2f}\t".format(psnr_result.item()))
        f.write("SSIM: {:.2f}\t".format(ssim_result.item()))
        f.write("FID: {:.2f}\n".format(FID))
        
        
    print(filename)
    print("With torch metrics PSNR:", round(psnr_result.item(), 2), 'SSIM', round(ssim_result.item(), 2))
    print("With Sitk          PSNR:", round(PSNR, 2), 'SSIM', round(SSIM, 2))
    print("FID分數:", round(FID, 2))
    print("==============================================")
    






    # ---------------------------------------------------------------------------------------------


if __name__ == '__main__':
    opt = TestOptions().parse()
    os.makedirs(opt.result, exist_ok=True)

    # 樣本路徑
    calculate_path = r'/home/po/Desktop/PDData/PD_result_images/'
    # 分離多巴胺發光體圖像路徑
    result_path = r'/home/po/Desktop/PDData/result/'
    # 分析數值文字文件路徑
    txt_path = r'/home/po/Desktop/PDData/result/results_PD_cycAB10.txt'


    test_set = NifitDataSet(calculate_path, transforms=None, which_direction='AtoB', train=False, test=True)

    data_len = len(test_set)
    avg_psnr = np.zeros(len(test_set))
    avg_ssim = np.zeros(len(test_set))
    avg_fid = np.zeros(len(test_set))

    for i in range(data_len):
        image = test_set.images_list[i]
        label = test_set.labels_list[i]

        filename = image
        # result = os.path.join(opt.result, filename)
        inference(image, label, filename, txt_path, result_path, i)

        avg_psnr[i] = PSNR
        avg_ssim[i] = SSIM
        avg_fid[i] = FID

    avg_psnr = np.mean(avg_psnr)
    avg_ssim = np.mean(avg_ssim)
    avg_fid = np.mean(avg_fid)
    
    with open(txt_path, "a") as f:
        f.write("=======================\n")
        f.write("AVG_PSNR: {:.2f}\t".format(avg_psnr))
        f.write("AVG_SSIM: {:.2f}\t".format(avg_ssim))
        f.write("AVG_FID: {:.2f}\n".format(avg_fid))
    print("AVG_PSNR:", round(avg_psnr, 3), "AVG_SSIM:", round(avg_ssim, 3), "AVG_FID:", round(avg_fid, 3))
    
    
