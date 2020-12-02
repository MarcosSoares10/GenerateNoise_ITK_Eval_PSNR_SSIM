import SimpleITK as sitk
import numpy as np
import math
import cv2


path_image = "IMGS/CT"
extension = ".nii"
stdlist = [100,90,80,70,60,50,40,30,20,10]



'''
    this function is used to compute psnr
input:
    ct_generated and ct_groundtruth
output:
    psnr
'''
def psnr(ct_generated,ct_GT):
    ct_generated = sitk.GetArrayFromImage(ct_generated)
    ct_GT = sitk.GetArrayFromImage(ct_GT)
    ct_generated = ct_generated.astype(np.float64)
    ct_GT = ct_GT.astype(np.float64)

    mse = np.mean((ct_generated - ct_GT)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse)),mse



'''calculate SSIM
    img1, img2: [0, 255]
'''
def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = sitk.GetArrayFromImage(img1)
    img2 = sitk.GetArrayFromImage(img2)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


initial_img = sitk.ReadImage(path_image+extension)

for std in stdlist:
    print("STD: ",str(std))    
    noise_image = sitk.AdditiveGaussianNoise(initial_img,std) #AdditiveGaussianNoise(Image image1, double standardDeviation=1.0, double mean=0.0)
    PSNR,MSE = psnr(noise_image,initial_img)
    SSIM = round(ssim(noise_image,initial_img),2)
    print("PSNR: ",str(PSNR))
    print("SSIM: ",str(SSIM))
    sitk.WriteImage(noise_image,path_image+"_STD_"+str(std)+"_MSE_"+str(round(MSE,2))+"_PSNR_"+str(round(PSNR,2))+"_SSIM_"+str(SSIM)+"_"+extension)





#image = sitk.Cast(image, sitk.sitkFloat32 )
#image = sitk.RescaleIntensity(image,0.0,512)
#image_data = sitk.GetArrayFromImage(image)
#image_data = sitk.GetArrayFromImage(noise_image)
#plt.imshow(image_data[90,:,:],cmap='gray')
#plt.show()
