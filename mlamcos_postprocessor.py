from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy import linalg
from sklearn.covariance import (empirical_covariance, ledoit_wolf,
                                shrunk_covariance)
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor
from .info import num_classes_dict

import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
import io
from torchvision.transforms import v2
from sklearn.metrics import roc_auc_score


class CMCPostprocessor(BasePostprocessor):
    def __init__(self, config):
        self.config = config
        self.postprocessor_args = config.postprocessor.postprocessor_args
        self.magnitude = self.postprocessor_args.noise
        self.selected_layers = self.postprocessor_args.selected_layers
        self.feature_type_list = self.postprocessor_args.feature_type_list
        self.noise_type = self.postprocessor_args.noise_type
        self.topk = self.postprocessor_args.topk

        self.num_classes = num_classes_dict[self.config.dataset.name]
        self.num_layer = len(self.selected_layers)

        self.feature_mean, self.feature_prec = None, None
        self.alpha_list = None
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            # step 1: estimate initial mean and variance from training set
            self.feature_mean, self.feature_prec, self.transform_matrix, self.all_traces = \
                get_MDS_stat(net, id_loader_dict['train'], self.num_classes,
                             self.selected_layers, self.feature_type_list)
            # exit()
            # step 2: input process and hyperparam searching for alpha
            if self.postprocessor_args.alpha_list:
                print('\n Load predefined alpha list...')
                self.alpha_list = self.postprocessor_args.alpha_list
            else:
                print('\n Searching for optimal alpha list...')
                self.alpha_list = []
                # get in-distribution scores
                for layer_index in range(self.num_layer):
                    weighted_maha_cosine_scores = get_Mahalanobis_scores(
                        net, id_loader_dict['val'], self.num_classes,
                        self.feature_mean, self.feature_prec, self.all_traces,
                        self.transform_matrix, layer_index, self.selected_layers,
                        self.feature_type_list, self.noise_type, self.magnitude)

                    self.alpha_list.append(weighted_maha_cosine_scores)

            self.alpha_list = torch.tensor(self.alpha_list)
            _, indices = torch.topk(-self.alpha_list, self.num_layer - self.topk)
            self.alpha_list[indices] = -float('inf')
            self.alpha_list = torch.nn.functional.softmax(self.alpha_list, dim = -1).unsqueeze(1)
            self.setup_flag = True
            print('Optimal alpha list: ', self.alpha_list)
        else:
            pass

    def postprocess(self, net: nn.Module, data: Any):
        for layer_index in range(self.num_layer):

            pred, score = compute_Mahalanobis_score(net,
                                                    Variable(
                                                        data,
                                                        requires_grad=True),
                                                    self.num_classes,
                                                    self.feature_mean,
                                                    self.feature_prec,
                                                    self.all_traces,
                                                    self.transform_matrix,
                                                    layer_index,
                                                    self.selected_layers,
                                                    self.feature_type_list,
                                                    self.noise_type,
                                                    self.magnitude,
                                                    return_pred=True)
            if layer_index == 0:
                score_list = score.view([-1, 1])
            else:
                score_list = torch.cat((score_list, score.view([-1, 1])), 1)
        alpha = torch.cuda.FloatTensor(self.alpha_list.cuda())
        conf = torch.matmul(score_list, alpha)
        return pred, conf

    def set_hyperparam(self, hyperparam: list):
        self.magnitude = hyperparam[0]

    def get_hyperparam(self):
        return self.magnitude


def tensor2list(x):
    return x.data.cpu().tolist()


def get_torch_feature_stat(feature, only_mean=False):
    feature = feature.view([feature.size(0), feature.size(1), -1])
    feature_mean = torch.mean(feature, dim=-1)
    feature_var = torch.var(feature, dim=-1)
    if feature.size(-2) * feature.size(-1) == 1 or only_mean:
        # [N, C, 1, 1] does not need variance for kernel
        feature_stat = feature_mean
    else:
        feature_stat = torch.cat((feature_mean, feature_var), 1)
    return feature_stat


def process_feature_type(feature_temp, feature_type):
    if feature_type == 'flat':
        feature_temp = feature_temp.view([feature_temp.size(0), -1])
    elif feature_type == 'stat':
        feature_temp = get_torch_feature_stat(feature_temp)
    elif feature_type == 'mean':
        feature_temp = get_torch_feature_stat(feature_temp, only_mean=True)
    else:
        raise ValueError('Unknown feature type')
    return feature_temp


def reduce_feature_dim(feature_list_full, label_list_full, feature_process):
    if feature_process == 'none':
        transform_matrix = np.eye(feature_list_full.shape[1])
    else:
        feature_process, kept_dim = feature_process.split('_')
        kept_dim = int(kept_dim)
        if feature_process == 'capca':
            lda = InverseLDA(solver='eigen')
            lda.fit(feature_list_full, label_list_full)
            transform_matrix = lda.scalings_[:, :kept_dim]
        elif feature_process == 'pca':
            pca = PCA(n_components=kept_dim)
            pca.fit(feature_list_full)
            transform_matrix = pca.components_.T
        elif feature_process == 'lda':
            lda = LinearDiscriminantAnalysis(solver='eigen')
            lda.fit(feature_list_full, label_list_full)
            transform_matrix = lda.scalings_[:, :kept_dim]
        else:
            raise Exception('Unknown Process Type')
    return transform_matrix

def get_layer_weight(layer_index, all_traces):
    min_t = min(all_traces)
    current_t = all_traces[layer_index]
    
    # 使用指数衰减：迹越大，lambda 越小
    # lamda 会落在 (0, 1] 区间
    lamda = torch.exp(-(current_t / min_t))
    return lamda

@torch.no_grad()
def get_MDS_stat(model, train_loader, num_classes, selected_layer, feature_type_list):
    model.eval()
    device = next(model.parameters()).device
    num_layer = len(selected_layer)
    
    # Initialize trackers
    class_counts = torch.zeros(num_layer, num_classes, device=device)
    feature_sums = [None] * num_layer 
    
    # --- Pass 1: Compute Means (One pass for all layers) ---
    for batch in tqdm(train_loader, desc='Pass 1: Means'):
        data = batch['data_aux'].to(device)
        labels = batch['label'].to(device)
        
        _, feature_list = model(data, return_feature_list=True)
        # feature_list.append(_.view(_.shape[0], _.shape[1], 1, 1))

        for layer_idx in range(num_layer):
            f = process_feature_type(feature_list[selected_layer[layer_idx]], 
                                     feature_type_list[layer_idx])
            
            if feature_sums[layer_idx] is None:
                feature_sums[layer_idx] = torch.zeros(num_classes, f.shape[1], device=device)

            for c in range(num_classes):
                mask = (labels == c)
                if mask.any():
                    feature_sums[layer_idx][c] += f[mask].sum(0)
                    class_counts[layer_idx][c] += mask.sum()

    feature_mean_list = [feature_sums[l] / class_counts[l].unsqueeze(1).clamp(min=1) 
                         for l in range(num_layer)]
    
    # --- Pass 2: Compute Precision (Batch loop OUTSIDE) ---
    # Initialize covariance matrices for all layers
    pooled_cov_list = [torch.zeros(m.shape[1], m.shape[1], device=device) 
                       for m in feature_mean_list]

    for batch in tqdm(train_loader, desc='Pass 2: Precision'):
        data = batch['data_aux'].to(device)
        labels = batch['label'].to(device)
        
        _, feature_list = model(data, return_feature_list=True)
        # feature_list.append(_.view(_.shape[0], _.shape[1], 1, 1))
        
        for layer_idx in range(num_layer):
            f = process_feature_type(feature_list[selected_layer[layer_idx]], 
                                     feature_type_list[layer_idx])
            
            for c in range(num_classes):
                mask = (labels == c)
                if mask.any():
                    # x - mu_c
                    centered_f = f[mask] - feature_mean_list[layer_idx][c]
                    # Update the shared covariance for this layer
                    pooled_cov_list[layer_idx] += torch.mm(centered_f.t(), centered_f)

    # Final Inversion
    precision_list = []
    all_traces = []
    for layer_idx in range(num_layer):
        n_samples = class_counts[layer_idx].sum()
        cov = pooled_cov_list[layer_idx] / n_samples
        
        # Regularize and Invert
        eye = torch.eye(cov.shape[0], device=device)
        precision = torch.inverse(cov + 1e-4 * eye)
        precision_list.append(precision)
        trace = torch.trace(torch.sqrt(cov))/cov.shape[0]
        all_traces.append(trace)
        print(f'Trace of layer {layer_idx} is: {trace}')

    return feature_mean_list, precision_list, None, all_traces

def calculate_maha_cosine_similarity(out_features, precision, traces, sample_mean, num_classes, layer_index, use_lamda):
    # Precision Matrix
    M_orig = precision[layer_index] 
    D = M_orig.shape[0]
    
    # Weighted Precision Matrix
    if use_lamda:
        lamda = get_layer_weight(layer_index, traces)
    else:
        lamda = 1
    identity = torch.eye(D).to(M_orig.device)
    M = lamda * M_orig + (1 - lamda) * identity
    
    f = out_features.data # (N, D)

    maha_cosine_score = 0

    for i in range(num_classes):
        mu = sample_mean[layer_index][i]
        
        # Calculate numerator
        M_mu = torch.mv(M, mu) # (D,)
        numerator = torch.mv(f, M_mu) # (N,)) # (N,)
        
        # Calculate denominator
        f_M_f = torch.diag(torch.mm(torch.mm(f, M), f.t())) # (N,)
        norm_f = torch.sqrt(torch.clamp(f_M_f, min=1e-9))

        mu_M_mu = torch.dot(mu, M_mu) # Scalar
        norm_mu = torch.sqrt(torch.clamp(mu_M_mu, min=1e-9))
        
        term_cos = numerator / (norm_f * norm_mu)

        if i == 0:
            maha_cosine_score = term_cos.view(-1, 1)
        else:
            maha_cosine_score = torch.cat((maha_cosine_score, term_cos.view(-1, 1)), 1)

    return maha_cosine_score

def get_Mahalanobis_scores(model, test_loader, num_classes, sample_mean,
                           precision, traces, transform_matrix, layer_index, selected_layers,
                           feature_type_list, noise_type, magnitude):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index
    '''
    model.eval()

    weighted_maha_cosine_scores = []
    for batch in tqdm(test_loader,
                      desc=f'{test_loader.dataset.name}_layer{layer_index}'):

        weighted_maha_cosine_score = compute_Mahalanobis_score(
            model, batch, num_classes, sample_mean, precision, traces, transform_matrix,
            layer_index, selected_layers, feature_type_list, noise_type, magnitude)
        
        weighted_maha_cosine_scores.append(weighted_maha_cosine_score)
    
    weighted_maha_cosine_scores = torch.cat(weighted_maha_cosine_scores, dim = 0)
    weighted_maha_cosine_scores = weighted_maha_cosine_scores.sum().item()/len(weighted_maha_cosine_scores)
    
    # return maha_structure_corrs, euclidean_structure_corrs, manhattan_structure_corrs, cosine_structure_corrs, feature_corrs
    return weighted_maha_cosine_scores

def get_forced_different_indices(labels):
    """
    Returns indices such that labels[i] != labels[indices[i]]
    """
    B = labels.size(0)
    # 1. Start with a simple cyclic shift of 1
    indices = (torch.arange(B) + 1) % B
    
    # 2. Check for collisions (where labels are the same)
    # This usually only happens in small batches or very imbalanced data
    for offset in range(2, B):
        mask = (labels == labels[indices]).cpu()
        if not mask.any():
            break
        # Shift only the colliding ones further
        indices[mask] = (indices[mask] + 1) % B
        
    return indices

def compute_Mahalanobis_score(model,
                              data,
                              num_classes,
                              sample_mean,
                              precision,
                              traces,
                              transform_matrix,
                              layer_index,
                              selected_layers,
                              feature_type_list,
                              noise_type,
                              magnitude,
                              return_pred=False):
    
    if return_pred == False:
        label = data['label'].cuda()
        data = data['data'].cuda()
        data = Variable(data, requires_grad=True)
    
    # extract features
    batch_size = data.shape[0]
    logits, out_features = model(data, return_feature_list=True)

    out_features.append(logits.view(logits.shape[0], logits.shape[1], 1, 1))
    out_features = process_feature_type(out_features[selected_layers[layer_index]],
                                        feature_type_list[layer_index])
    if transform_matrix is not None:
        out_features = torch.mm(out_features, transform_matrix[layer_index])

    # compute maha_cosine similarity score
    weighted_maha_cosine_score = calculate_maha_cosine_similarity(out_features, precision, traces, sample_mean, num_classes, layer_index, use_lamda = True).detach().cpu()

    # Inference
    sample_pred = logits.argmax(1)
    if return_pred:
        return sample_pred, weighted_maha_cosine_score.max(1)[0].cuda()

    # Noised Input_processing
    if noise_type == 'cutmix-center':
        """
        Applies CutMix by replacing ONLY the center 30% of the image 
        with the corresponding patch from a shuffled batch.
        """
        data = data.detach()
        indices = get_forced_different_indices(label)
        shuffled_data = data[indices]

        # Get dimensions: (H, W) or (D, H, W)
        dims = data.shape[2:]
        num_spatial_dims = len(dims)
        
        # Calculate start and end indices for the center crop
        # Formula: Start = 0.5 - (mag/2), End = 0.5 + (mag/2)
        m_starts = [int(s * (0.5 - magnitude / 2)) for s in dims]
        m_ends = [int(s * (0.5 + magnitude / 2)) for s in dims]

        if num_spatial_dims == 2:
            ms_h, me_h = m_starts[0], m_ends[0]
            ms_w, me_w = m_starts[1], m_ends[1]
            
            # Mix the center rectangle
            data[:, :, ms_h:me_h, ms_w:me_w] = shuffled_data[:, :, ms_h:me_h, ms_w:me_w]

        elif num_spatial_dims == 3:
            ms_d, me_d = m_starts[0], m_ends[0]
            ms_h, me_h = m_starts[1], m_ends[1]
            ms_w, me_w = m_starts[2], m_ends[2]
            
            # Mix the center cube
            data[:, :, ms_d:me_d, ms_h:me_h, ms_w:me_w] = shuffled_data[:, :, ms_d:me_d, ms_h:me_h, ms_w:me_w]
    
    elif noise_type == 'cutmix':

        """
        Applies CutMix to 4D [B, C, H, W] or 5D [B, C, D, H, W] tensors.
        Returns only the mixed image batch.
        """

        data = data.detach()
        indices = get_forced_different_indices(label)
        shuffled_data = data[indices]

        # Get dimensions
        dims = data.shape[2:]  # (H, W) or (D, H, W)
        num_spatial_dims = len(dims)
        
        # Calculate crop boundaries for H and W (and D if 5D)
        # These will be the same for all corners
        crop_dims = [int(s * magnitude) for s in dims]
        
        # helper for limits
        limits = [s - c for s, c in zip(dims, crop_dims)]

        # We apply 4 corner replacements. 
        # For 5D, we usually apply this to the last two dims (H, W) 
        # and keep the full Depth (D), or apply to the 'corners' of the volume.
        
        if num_spatial_dims == 2:
            h_c, w_c = crop_dims
            h_lim, w_lim = limits
            
            # 1. Top-Left
            data[:, :, :h_c, :w_c] = shuffled_data[:, :, :h_c, :w_c]
            # 2. Top-Right
            data[:, :, :h_c, w_lim:] = shuffled_data[:, :, :h_c, w_lim:]
            # 3. Bottom-Left
            data[:, :, h_lim:, :w_c] = shuffled_data[:, :, h_lim:, :w_c]
            # 4. Bottom-Right
            data[:, :, h_lim:, w_lim:] = shuffled_data[:, :, h_lim:, w_lim:]
                    
        elif num_spatial_dims == 3:
            d_s, h_c, w_c = crop_dims
            d_lim, h_lim, w_lim = limits
            
            # In 3D/5D, we replace the corners of the H-W planes across ALL depth, 
            # OR the corners of the cube. Usually, researchers do the corners of the H-W plane.
            # Here is the "Cube Corner" version:
            data[:, :, :d_s, :h_c, :w_c] = shuffled_data[:, :, :d_s, :h_c, :w_c]
            data[:, :, :d_s, :h_c, w_lim:] = shuffled_data[:, :, :d_s, :h_c, w_lim:]
            data[:, :, :d_s, h_lim:, :w_c] = shuffled_data[:, :, :d_s, h_lim:, :w_c]
            data[:, :, :d_s, h_lim:, w_lim:] = shuffled_data[:, :, :d_s, h_lim:, w_lim:]
            # Note: To be perfectly symmetric in 3D, you'd do 8 corners.


    #  pred of noised input
    with torch.no_grad():
        _, noise_out_features = model(Variable(data),
                                      return_feature_list=True)
        noise_out_features.append(_.view(_.shape[0], _.shape[1], 1, 1))
        noise_out_features = process_feature_type(
            noise_out_features[selected_layers[layer_index]], feature_type_list[layer_index])
        if transform_matrix is not None:
            noise_out_features = torch.mm(noise_out_features,
                                        transform_matrix[layer_index])

    # noised compute maha_cosine similarity score
    noised_weighted_maha_cosine_score = calculate_maha_cosine_similarity(noise_out_features, precision, traces, sample_mean, num_classes, layer_index, use_lamda = True).detach().cpu()

    # Compare the max Mahalanobis cosine score of the original and noised inputs
    noised_weighted_maha_cosine_score = noised_weighted_maha_cosine_score.max(1)[0]
    weighted_maha_cosine_score = weighted_maha_cosine_score.max(1)[0]
    # If the noised score is less than the original score, it suggests that the noise has successfully reduced the confidence, which is a sign of a correct prediction.
    weighted_maha_cosine_corr = (noised_weighted_maha_cosine_score < weighted_maha_cosine_score)

    return weighted_maha_cosine_corr
