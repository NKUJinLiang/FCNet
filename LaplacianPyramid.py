import cv2 as cv
import time
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from skimage.io import imread, imshow
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch

class Lap_Pyramid(nn.Module):
    def __init__(self, num_high=3):
        super(Lap_Pyramid, self).__init__()

        self.num_high = num_high
        self.kernel = self.gauss_kernel()

    def gauss_kernel(self, device=torch.device('cuda'), channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[-1]
        pyramid = []
        pyramid.append(image)
        for level in reversed(pyr[:-1]):
            up = self.upsample(image)
            if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
                up = nn.functional.interpolate(up, size=(level.shape[2], level.shape[3]))
            image = up + level
            pyramid.append(image)
        return pyramid

    def pyramid_loss(self, pyr, gt):
        loss = 0
        image_1 = pyr[-1]
        image_2 = gt[-1]

        list1 = []
        list2 = []

        for level in reversed(pyr[:-1]):
            up = self.upsample(image_1)
            if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
                up = nn.functional.interpolate(up, size=(level.shape[2], level.shape[3]))
            image_1 = up + level
            list1.append(image_1)

        for level in reversed(gt[:-1]):
            up = self.upsample(image_2)
            if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
                up = nn.functional.interpolate(up, size=(level.shape[2], level.shape[3]))
            image_2 = up + level
            list2.append(image_2)

        for step in range(len(list1)):
            loss = loss + (len(list1)-step)*(len(list1)-step)*torch.mean(4 * torch.sum(torch.abs(list1[step] - list2[step]), [1, 2, 3]), dim=0)

        return loss

#image = imread("images/" + "512512.jpg")
#print(image.shape)
#print(lapalian_demo(image)[3].shape)


if __name__ == "__main__":
    #x_0 = torch.rand(size=(2,3,3))

    x_1 = torch.rand(size=(7, 3, 75, 113)).cuda()

    x_2 = torch.rand(size=(7, 3, 150, 226)).cuda()
    x_3 = torch.rand(size=(7, 3, 300, 452)).cuda()
    x_4 = torch.rand(size=(7, 3, 600, 903)).cuda()
    x_5 = torch.rand(size=(7, 3, 1200, 1200)).cuda()

    image = imread("../testimage/1/" + "1.jpg")
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)
    image = image.cuda()
    lp = Lap_Pyramid()
    pry1 = lp.pyramid_decom(image)
    pry2 = lp.pyramid_recons(pry1)

    for index in range(len(pry2)):
        torchvision.utils.save_image(pry2[index], "pry2"+str(index) + ".jpg")


    for index in range(len(pry1)):
        torchvision.utils.save_image(pry1[index], "pry1"+str(index) + ".jpg")





'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = LPTNPaper()
image = imread("../testimage/1/" + "1.jpg")
image = transforms.ToTensor()(image)
image = image.unsqueeze(0)
image = image.cuda()
lp = Lap_Pyramid_Conv(3)

model = LPTNPaper()

start = time.time()

list = lp.pyramid_decom(img=image)
end_time = (time.time() - start)
print(end_time)
index = 0
for img in list:
    torchvision.utils.save_image(img, str(index) + ".jpg")
    index = index + 1

image = lp.pyramid_recons(list)


torchvision.utils.save_image(image, "str(index)" + ".jpg")

#print(img_Image)
#print(img2)
'''