import os
from options.test_options import TestOptions
from utils.NiftiDataset import *
import utils.NiftiDataset as NiftiDataset_testing
from torch.utils.data import DataLoader
from models import create_model
import math
from torch.autograd import Variable
from tqdm import tqdm
import datetime
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import torch
import numpy as np

from scipy import linalg
# ===========================================================
from torch.nn.functional import adaptive_avg_pool2d
from torch.nn.functional import mse_loss

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
    model.eval()
    act = []
    # pred = torch.zeros([images.shape[2], 1000])
    with torch.no_grad():
        for start_idx in range(0, images.shape[2]):
            batch_var = images[:, :, start_idx, :, :]

            if batch_var.shape[1] == 1:
                batch_var = batch_var.repeat(1, 3, 1, 1)
            if batch_var.shape[1] != 3:
                raise ValueError('Input images should be either grayscale or RGB')

            pred = model(batch_var)[0]
            pred = torch.reshape(pred, (pred.shape[0], 1, 1))
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1)).squeeze(-1).squeeze(-1)
            act.append(pred.cpu().detach().numpy())
    act = torch.from_numpy(np.concatenate(act, axis=0)).double()
    mu = torch.mean(act, dim=0)
    sigma = np.cov(act.numpy(), rowvar=False)
    return mu, sigma

def calculate_fid(images1, images2, model, batch_size=104):
    """計算FID"""
    images1 = images1.astype('float32')
    images2 = images2.astype('float32')
    images1 = torch.tensor(images1)
    images2 = torch.tensor(images2)
    mu1, sigma1 = calculate_activation_statistics(images1, model, batch_size)
    mu2, sigma2 = calculate_activation_statistics(images2, model, batch_size)
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid

# 載入預訓練的3D影像分類模型
model_inception_v3 = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
# ===========================================================





def from_numpy_to_itk(image_np, image_itk):
    image_np = np.transpose(image_np, (2, 1, 0))
    image = sitk.GetImageFromArray(image_np)
    image.SetOrigin(image_itk.GetOrigin())
    image.SetDirection(image_itk.GetDirection())
    image.SetSpacing(image_itk.GetSpacing())
    return image


def prepare_batch(image, ijk_patch_indices):
    image_batches = []
    for batch in ijk_patch_indices:
        image_batch = []
        for patch in batch:
            image_patch = image[patch[0]:patch[1], patch[2]:patch[3], patch[4]:patch[5]]
            image_batch.append(image_patch)

        image_batch = np.asarray(image_batch)
        # image_batch = image_batch[:, :, :, :, np.newaxis]
        image_batches.append(image_batch)

    return image_batches

def inference(model, image_path, label_path, result_path, resample, resolution, patch_size_x,
              patch_size_y, patch_size_z, stride_inplane, stride_layer, batch_size=1):

    # create transformations to image and labels
    transforms1 = [
        NiftiDataset_testing.Resample(resolution, resample)
    ]

    transforms2 = [
        NiftiDataset_testing.Padding((patch_size_x, patch_size_y, patch_size_z))
    ]

    # read image file
    reader = sitk.ImageFileReader()
    reader.SetFileName(image_path)
    image = reader.Execute()

    reader.SetFileName(label_path)
    real = reader.Execute()

    # normalize the image
    image = Normalization(image)
    real = Normalization(real)

    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
    image = castImageFilter.Execute(image)
    real = castImageFilter.Execute(real)

    # create empty label in pair with transformed image
    label_tfm = sitk.Image(image.GetSize(), sitk.sitkFloat32)

    label_tfm.SetOrigin(image.GetOrigin())
    label_tfm.SetDirection(image.GetDirection())
    label_tfm.SetSpacing(image.GetSpacing())



    sample = {'image': image, 'label': label_tfm}

    for transform in transforms1:
        sample = transform(sample)
        # real = transform(real)

    # keeping track on how much padding will be performed before the inference
    image_array = sitk.GetArrayFromImage(sample['image'])

    pad_x = patch_size_x - (patch_size_x - image_array.shape[2])
    pad_y = patch_size_y - (patch_size_y - image_array.shape[1])
    pad_z = patch_size_z - (patch_size_z - image_array.shape[0])

    image_pre_pad = sample['image']

    for transform in transforms2:
        sample = transform(sample)

    image_tfm, label_tfm = sample['image'], sample['label']

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(real.GetSpacing())
    resampler.SetSize((patch_size_x, patch_size_y, patch_size_z))
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetOutputOrigin(real.GetOrigin())
    resampler.SetOutputDirection(real.GetDirection())
    real = resampler.Execute(real)

    # convert image to numpy array
    image_np = sitk.GetArrayFromImage(image_tfm)
    label_np = sitk.GetArrayFromImage(label_tfm)
    real_np = sitk.GetArrayFromImage(real)

    label_np = np.asarray(label_np, np.float32)
    real_np = np.asarray(real_np, np.float32)
    # unify numpy and sitk orientation
    image_np = np.transpose(image_np, (2, 1, 0))
    real_np = np.transpose(real_np, (2, 1, 0))
    label_np = np.transpose(label_np, (2, 1, 0))

    # ----------------- Padding the image if the z dimension still is not even ----------------------

    if (image_np.shape[2] % 2) == 0:
        Padding = False
    else:
        image_np = np.pad(image_np, ((0, 0), (0, 0), (0, 1)), 'edge')
        label_np = np.pad(label_np, ((0, 0), (0, 0), (0, 1)), 'edge')
        real_np = np.pad(real_np, ((0, 0), (0, 0), (0, 1)), 'edge')
        Padding = True

    # ------------------------------------------------------------------------------------------------

    # a weighting matrix will be used for averaging the overlapped region
    weight_np = np.zeros(label_np.shape)

    # prepare image batch indices
    inum = int(math.ceil((image_np.shape[0] - patch_size_x) / float(stride_inplane))) + 1
    jnum = int(math.ceil((image_np.shape[1] - patch_size_y) / float(stride_inplane))) + 1
    knum = int(math.ceil((image_np.shape[2] - patch_size_z) / float(stride_layer))) + 1

    patch_total = 0
    ijk_patch_indices = []
    ijk_patch_indicies_tmp = []

    for i in range(inum):
        for j in range(jnum):
            for k in range(knum):
                if patch_total % batch_size == 0:
                    ijk_patch_indicies_tmp = []

                istart = i * stride_inplane
                if istart + patch_size_x > image_np.shape[0]:  # for last patch
                    istart = image_np.shape[0] - patch_size_x
                iend = istart + patch_size_x

                jstart = j * stride_inplane
                if jstart + patch_size_y > image_np.shape[1]:  # for last patch
                    jstart = image_np.shape[1] - patch_size_y
                jend = jstart + patch_size_y

                kstart = k * stride_layer
                if kstart + patch_size_z > image_np.shape[2]:  # for last patch
                    kstart = image_np.shape[2] - patch_size_z
                kend = kstart + patch_size_z

                ijk_patch_indicies_tmp.append([istart, iend, jstart, jend, kstart, kend])

                if patch_total % batch_size == 0:
                    ijk_patch_indices.append(ijk_patch_indicies_tmp)

                patch_total += 1

    batches = prepare_batch(image_np, ijk_patch_indices)
    real_batches = prepare_batch(real_np, ijk_patch_indices)

    # =====================================================================

    # =====================================================================
    for i in tqdm(range(len(batches))):
        batch = batches[i]

        batch = (batch - 127.5) / 127.5


        batch = torch.from_numpy(batch[np.newaxis, :, :, :])
        batch = Variable(batch.cuda())

        # pred = model(batch)
        model.set_input(batch)
        model.test()
        pred = model.get_current_visuals()
        pred = pred['fake_B']




        # pred = pred.squeeze().data.cpu().numpy().astype(np.float32)
        #
        # fake = pred
        # target = np.array(real_np)
        # target_max = np.max(target)
        # target_min = np.min(target)


        # target = (target / (target_max - target_min)) * 255
        # target = target.squeeze().astype(np.uint8)
        # pred = (pred * 127.5) + 127.5

        # global PSNR
        # global SSIM
        # global FID
        #
        # PSNR = psnr(target, pred.astype(np.uint8))
        # SSIM = ssim(target, pred.astype(np.uint8))

        # 計算FID分數
        # fid_pred = np.expand_dims(pred.astype(np.float32), axis=(0, 1))
        # fid_target = np.expand_dims(target.astype(np.float32), axis=(0, 1))
        # FID = calculate_fid(fid_target, fid_pred, model_inception_v3, batch_size=1)
        # print("FID分數:", FID)




        istart = ijk_patch_indices[i][0][0]
        iend = ijk_patch_indices[i][0][1]
        jstart = ijk_patch_indices[i][0][2]
        jend = ijk_patch_indices[i][0][3]
        kstart = ijk_patch_indices[i][0][4]
        kend = ijk_patch_indices[i][0][5]
        #label_np[istart:iend, jstart:jend, kstart:kend] += pred[:, :, :]
        label_np[istart:iend, jstart:jend, kstart:kend] += np.reshape(pred[:,:,:].cpu().numpy(), label_np[istart:iend, jstart:jend, kstart:kend].shape)
        weight_np[istart:iend, jstart:jend, kstart:kend] += 1.0



    print("{}: Evaluation complete".format(datetime.datetime.now()))



    # eliminate overlapping region using the weighted value
    label_np = (np.float32(label_np) / np.float32(weight_np) + 0.01)

    # removed the 1 pad on z
    if Padding is True:
        label_np = label_np[:, :, 0:(label_np.shape[2] - 1)]

    # removed all the padding
    label_np = label_np[:pad_x, :pad_y, :pad_z]
    real_np = real_np[:pad_x, :pad_y, :pad_z]


    # fig = plt.figure()
    # plt.plot(ironman, np.sin(ironman), '.')


    # convert back to sitk space
    label = from_numpy_to_itk(label_np, image_pre_pad)
    # ---------------------------------------------------------------------------------------------
    # print("PSNR:", round(PSNR, 2), 'SSIM', round(SSIM, 2))
    # save label
    writer = sitk.ImageFileWriter()

    if resample is True:
        print("{}: Resampling label back to original image space...".format(datetime.datetime.now()))
        # label = resample_sitk_image(label, spacing=image.GetSpacing(), interpolator='bspline')   # keep this commented

        label = resize(label, (sitk.GetArrayFromImage(image)).shape[::-1], sitk.sitkLinear)
        label.SetDirection(image.GetDirection())
        label.SetOrigin(image.GetOrigin())
        label.SetSpacing(image.GetSpacing())
    else:
        label = label

    writer.SetFileName(result_path)
    writer.Execute(label)

    print("{}: Save evaluate label at {} success".format(datetime.datetime.now(), result_path))



if __name__ == '__main__':
    opt = TestOptions().parse()
    os.makedirs(opt.result, exist_ok=True)

    model = create_model(opt)
    model.setup(opt)



    test_set = NifitDataSet(opt.val_path, transforms=None, which_direction='AtoB',train=False, test=True)

    data_len = len(test_set)
    #avg_psnr = np.zeros(len(test_set))
    #avg_ssim = np.zeros(len(test_set))
    #avg_fid = np.zeros(len(test_set))


    for i in range(data_len):
        image = test_set.images_list[i]
        label = test_set.labels_list[i]
        filename = 'result_'+str(i)+'.nii'
        result = os.path.join(opt.result, filename)
        inference(model, image, label, result, opt.resample, opt.new_resolution, opt.patch_size[0],
                  opt.patch_size[1], opt.patch_size[2], opt.stride_inplane, opt.stride_layer, 1)

        # avg_psnr[i] = PSNR
        # avg_ssim[i] = SSIM
        # avg_fid[i] = FID

    # avg_psnr = np.mean(avg_psnr)
    # avg_ssim = np.mean(avg_ssim)
    # avg_fid = np.mean(avg_fid)

    # print("AVG_PSNR:", round(avg_psnr, 2), "AVG_SSIM:", round(avg_ssim, 2), "AVG_FID:", round(avg_fid, 2))

