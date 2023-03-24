# coding:utf-8
import lpips
import cv2
import glob
import tqdm

class util_of_lpips():
    def __init__(self, net='vgg', use_gpu=False):
        '''
        Parameters
        ----------
        net: str
            抽取特征的网络，['alex', 'vgg']
        use_gpu: bool
            是否使用GPU，默认不使用
        Returns
        -------
        References
        -------
        https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_2imgs.py

        '''
        ## Initializing the model
        self.loss_fn = lpips.LPIPS(net=net)
        self.use_gpu = use_gpu
        if use_gpu:
            self.loss_fn.cuda()

    def calc_lpips(self, img_p):
        '''
        Parameters
        ----------
        img1_path : str
            图像1的路径.
        img2_path : str
            图像2的路径.
        Returns
        -------
        dist01 : torch.Tensor
            学习的感知图像块相似度(Learned Perceptual Image Patch Similarity, LPIPS).

        References
        -------
        https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_2imgs.py

        '''
        img1_path, img2_path = img_p
        # Load images
        img0 = lpips.im2tensor(lpips.load_image(img1_path))  # RGB image from [-1,1]
        img1 = lpips.im2tensor(lpips.load_image(img2_path))

        if self.use_gpu:
            img0 = img0.cuda()
            img1 = img1.cuda()

        dist01 = self.loss_fn.forward(img0, img1).item()
        del img0
        del img1
        return dist01


if __name__ == '__main__':
    real_way = ''
    fake_Way = ''
    img_list_A = glob.glob(real_way + '/*')
    # img_list_B = glob.glob(fake_Way + '/*')
    img_list_A.sort()
    # img_list_B.sort()
    lpips_list = []
    # ssim_list = []
    LPIPS_Mec = util_of_lpips(use_gpu=True)
    for i in tqdm.tqdm(range(len(img_list_A))):

        # print(img_list_A[i].split('/')[-1], img_list_B[i].split('/')[-1])
        img_list_B_way = fake_Way + '/' + img_list_A[i].split('/')[-1]
        # img_list_B_way = glob.glob(fake_Way + '/' + img_list_A[i].split('/')[-1].split('.')[0]+'_*.png')[0]
        # realim = cv2.imread(img_list_A[i])
        # fakeim = cv2.imread(img_list_B_way)
        # iou_s = diceCoeff_score(outim, gtim)
        # print([img_list_A[i], img_list_B_way])
        lpips_value = LPIPS_Mec.calc_lpips([img_list_A[i], img_list_B_way])
        lpips_list.append(lpips_value)
        # ssim_list.append(ssim)
    lpips_avr = sum(lpips_list)/len(lpips_list)

    print('lpips_avr')
    # print(psnr_list)
    print(lpips_avr)
