import SimpleITK as sitk
import numpy as np
import cv2


path_image = "IMGA/CT"
extension = ".nii"
stdlist = [300,200,100,80,40]



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

    mse=np.sqrt(np.mean((ct_generated-ct_GT)**2))
    max_I=np.max([np.max(ct_generated),np.max(ct_GT)])
    return 20.0*np.log10(max_I/mse)



'''calculate SSIM
    img1, img2: [0, 255]
'''
def ssim(img1, img2):

    img1 = sitk.GetArrayFromImage(img1)
    img2 = sitk.GetArrayFromImage(img2)
    
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  
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



initial_img = sitk.ReadImage(path_image+extension)

for std in stdlist:
    
    noise_image = sitk.AdditiveGaussianNoise(initial_img,std) #AdditiveGaussianNoise(Image image1, double standardDeviation=1.0, double mean=0.0)
    PSNR = round(psnr(noise_image,initial_img),2)
    SSIM = round(ssim(noise_image,initial_img),2)
    print("STD: ",str(std))
    print("PSNR: ",str(PSNR))
    print("SSIM: ",str(SSIM))
    sitk.WriteImage(noise_image,path_image+"_PSNR_"+str(PSNR)+"_STD_"+str(std)+"_"+extension)














#image = sitk.Cast(image, sitk.sitkFloat32 )
#image = sitk.RescaleIntensity(image,0.0,512)
#image_data = sitk.GetArrayFromImage(image)
#image_data = sitk.GetArrayFromImage(noise_image)
#plt.imshow(image_data[90,:,:],cmap='gray')
#plt.show()