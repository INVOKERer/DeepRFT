import torch
import numpy as np

eps = 1e-10


# Cross-correlation matrix
def cross_correlation(H_fuse, H_ref):
    N_spectral = H_fuse.shape[1]

    # Rehsaping fused and reference data
    H_fuse_reshaped = H_fuse.view(N_spectral, -1)
    H_ref_reshaped = H_ref.view(N_spectral, -1)

    # Calculating mean value
    mean_fuse = torch.mean(H_fuse_reshaped, 1).unsqueeze(1)
    mean_ref = torch.mean(H_ref_reshaped, 1).unsqueeze(1)

    CC = torch.sum((H_fuse_reshaped - mean_fuse) * (H_ref_reshaped - mean_ref), 1) / (torch.sqrt(
        torch.sum((H_fuse_reshaped - mean_fuse) ** 2, 1) * torch.sum((H_ref_reshaped - mean_ref) ** 2, 1)) )

    CC = torch.mean(CC)
    # print(CC, H_fuse.max(), H_fuse.min())
    return CC


# Spectral-Angle-Mapper (SAM)
def SAM(H_fuse, H_ref):
    # Compute number of spectral bands
    N_spectral = H_fuse.shape[1]

    # Rehsaping fused and reference data
    H_fuse_reshaped = H_fuse.view(N_spectral, -1)
    H_ref_reshaped = H_ref.view(N_spectral, -1)
    N_pixels = H_fuse_reshaped.shape[1]

    # Calculating inner product
    inner_prod = torch.nansum(H_fuse_reshaped * H_ref_reshaped, 0)
    fuse_norm = torch.nansum(H_fuse_reshaped ** 2, dim=0).sqrt()
    ref_norm = torch.nansum(H_ref_reshaped ** 2, dim=0).sqrt()

    # Calculating SAM
    SAM = torch.rad2deg(torch.nansum(torch.acos(inner_prod / (fuse_norm * ref_norm))) / N_pixels)
    return SAM


# Root-Mean-Squared Error (RMSE)
def RMSE(H_fuse, H_ref):
    # Rehsaping fused and reference data
    # H_fuse_reshaped = H_fuse.view(-1)
    # H_ref_reshaped = H_ref.view(-1)
    N_spectral = H_fuse.shape[1]
    # print(N_spectral)
    # Reshaping images
    H_fuse_reshaped = H_fuse.view(N_spectral, -1)
    H_ref_reshaped = H_ref.view(N_spectral, -1)
    # Calculating RMSE
    RMSE = torch.sqrt(torch.nansum((H_ref_reshaped - H_fuse_reshaped) ** 2) / H_fuse_reshaped.shape[0])
    return RMSE


# Erreur Relative Globale Adimensionnelle De Synthèse (ERGAS)
def ERGAS(H_fuse, H_ref, beta=1):
    # Compute number of spectral bands
    N_spectral = H_fuse.shape[1]

    # Reshaping images
    H_fuse_reshaped = H_fuse.view(N_spectral, -1)
    H_ref_reshaped = H_ref.view(N_spectral, -1)
    N_pixels = H_fuse_reshaped.shape[1]

    # Calculating RMSE of each band
    rmse = torch.sqrt(torch.nansum((H_ref_reshaped - H_fuse_reshaped) ** 2, dim=1) / N_pixels)
    mu_ref = torch.mean(H_ref_reshaped, dim=1)

    # Calculating Erreur Relative Globale Adimensionnelle De Synthèse (ERGAS)
    ERGAS = 100 * (1 / beta ** 2) * torch.sqrt(torch.nansum(torch.div(rmse, mu_ref) ** 2) / N_spectral)
    return ERGAS


# Peak SNR (PSNR)
def PSNR(H_fuse, H_ref):
    # Compute number of spectral bands
    N_spectral = H_fuse.shape[1]
    # print(N_spectral)
    # Reshaping images
    H_fuse_reshaped = H_fuse.view(N_spectral, -1)
    H_ref_reshaped = H_ref.view(N_spectral, -1)

    # Calculating RMSE of each band
    rmse = torch.sqrt(torch.sum((H_ref_reshaped - H_fuse_reshaped) ** 2, dim=1) / H_fuse_reshaped.shape[1])

    # Calculating max of H_ref for each band
    max_H_ref, _ = torch.max(H_ref_reshaped, dim=1)

    # Calculating PSNR
    PSNR = torch.nansum(10 * torch.log10(torch.div(max_H_ref, rmse) ** 2)) / N_spectral

    return PSNR # , rmse