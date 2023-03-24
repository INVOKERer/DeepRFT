%% Restormer: Efficient Transformer for High-Resolution Image Restoration
%% Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
%% https://arxiv.org/abs/2111.09881

close all;clear all;

% datasets = {'GoPro'};
datasets = {'GoPro', 'HIDE'};
% datasets = {'HIDE'};
num_set = length(datasets);

for idx_set = 1:num_set
    % file_path = strcat('./results/', 'GoPro_DCTformer32_8gpu/', datasets{idx_set}, '/');
    file_path = strcat('/home/mxt/106-48t/personal_data/mxt/exp_results/cvpr2023/results/', 'GoPro_DCTformerPLUS_8gpu/', datasets{idx_set}, '/');
    gt_path = strcat('/home/mxt/106-48t/personal_data/mxt/Datasets/Deblur/', datasets{idx_set}, '/test/sharp/');
    path_list = [dir(strcat(file_path,'*.jpg')); dir(strcat(file_path,'*.png'))];
    gt_list = [dir(strcat(gt_path,'*.jpg')); dir(strcat(gt_path,'*.png'))];
    img_num = length(path_list);

    total_psnr = 0;
    total_ssim = 0;
    if img_num > 0 
        for j = 1:img_num 
           image_name = path_list(j).name;
           gt_name = gt_list(j).name;
           input = imread(strcat(file_path,image_name));
           gt = imread(strcat(gt_path, gt_name));
           ssim_val = ssim(input, gt);
           psnr_val = psnr(input, gt);
           total_ssim = total_ssim + ssim_val;
           total_psnr = total_psnr + psnr_val;
       end
    end
    qm_psnr = total_psnr / img_num;
    qm_ssim = total_ssim / img_num;
    
    fprintf('For %s dataset PSNR: %f SSIM: %f\n', datasets{idx_set}, qm_psnr, qm_ssim);

end
