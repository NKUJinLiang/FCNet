import torch
import torch.nn as nn
import cv2 as cv
import torchvision
import torch.optim
import os
from Unet import Fusion_Model
from torch.utils.data import DataLoader
from loaddata import ImageSeqDataset
from loaddata import image_seq_loader
from torchvision import transforms
from LaplacianPyramid import Lap_Pyramid
from batch_transformers import BatchRandomResolution, BatchToTensor, BatchRGBToYCbCr, YCbCrToRGB, BatchTestResolution
import pandas as pd
from skimage.io import imread, imshow
import time
from skimage.metrics import structural_similarity as ssim
from skimage.metrics.simple_metrics import peak_signal_noise_ratio as psnr

# import lpips
# loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
# loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

# snapshots-primary/Epoch199.pth
# snapshots-without-exp/Epoch199.pth
# snapshots-without-spa/Epoch199.pth

Train_transform = transforms.Compose([
    BatchToTensor(),
])


def Test(Test_root, label_path,epoch):
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    num_high = 3
    testmodel = Fusion_Model(num_high=3).cuda()
    testmodel.load_state_dict(torch.load('snapshots/'+epoch))

    time1 = 0
    evl1 = 0
    evl2 = 0
    evl_lpipsvgg = 0
    evl_lpipsalex = 0

    # 数据集处理
    train_transform = transforms.Compose([
        BatchToTensor(),
    ])

    # 数据集路径
    # datapath = "./data/"
    # 构建数据集

    train_data = ImageSeqDataset(csv_file=os.path.join(Test_root, 'test.txt'),
                                 Train_img_seq_dir=Test_root,
                                 Label_img_dir=label_path,
                                 Train_transform=train_transform,
                                 Label_transform=transforms.ToTensor(),
                                 randomlist=False)

    train_loader = DataLoader(train_data,
                              batch_size=1,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=1)

    for step, sample_batched in enumerate(train_loader):
        test_image, label_image = sample_batched['Train'], sample_batched['Lable']

        test_image = test_image.squeeze(0).cuda()
        #a = test_image[0, :, :, :]
        #test_image = test_image[3:5, :, :, :]
        #test_image = torch.cat([a.unsqueeze(0),test_image],0)
        
        print(test_image.shape)

        label_image = label_image.cuda()

        start = time.time()


        out1, out2, out3, out4 = testmodel(test_image)


        end_time = (time.time() - start)

        time1 = time1 + end_time

        average_time = time1 / (step + 1)

        result_path = "./" + epoch + "/fusionresults/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        torchvision.utils.save_image(out4, result_path + str(step + 1) + ".jpg")

        image = cv.imread(result_path + str(step + 1) + ".jpg")
        label = cv.imread("../testimage/label/" + str(step + 1) + ".jpg")

        print(image.shape)
        print(label.shape)

        evl1 = evl1 + ssim(image, label, multichannel=True)
        evl2 = evl2 + psnr(image, label)
        evl_ssim = evl1 / (step + 1)
        evl_psnr = evl2 / (step + 1)

        print("time", average_time)
        print("psnr", evl_psnr)
        print("ssim", evl_ssim)
        f = epoch + "fusion.txt"
        with open(f, "w") as file:  # ”w"代表着每次运行都覆盖内容
            file.write("ssim=" + str(evl_ssim) + "\n")
            file.write("psnr=" + str(evl_psnr) + "\n")


if __name__ == '__main__':
    test_path = '../testimage/'
    label_path = "../testimage/label/"
    epochlist = []
    #epochlist.append("Epoch40.pth")
    #epochlist.append("Epoch60.pth")
    #epochlist.append("Epoch80.pth")
    #epochlist.append("Epoch100.pth")
    #epochlist.append("Epoch120.pth")
    #epochlist.append("Epoch140.pth")
    epochlist.append("Epoch135.pth")
    #epochlist.append("Epoch155.pth")
    #epochlist.append("Epoch165.pth")
    #epochlist.append("Epoch180.pth")
    #epochlist.append("Epoch199.pth")
    for epoch in epochlist:
        Test(test_path, label_path, epoch)

