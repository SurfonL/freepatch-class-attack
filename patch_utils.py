# Adversarial Patch: patch_utils
# utils for patch initialization and mask generation
# Created by Junbo Zhao 2020/3/19

import numpy as np
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import os
import cv2
from PIL import Image

# Generate the mask and apply the patch
# TODO: Add circle type
def mask_generation(args, im_h, im_w):

    mask_dir = os.path.join(args.data_dir,'mask.jpg')
    mask = cv2.imread(mask_dir)
    mask = cv2.resize(mask, (im_w,im_h))
    
    #clean up possible interpolation errors due to resizing operation
    th = 200
    condition = [mask[:,:,0]<th,mask[:,:,1]<th,mask[:,:,2]<th]
    mask[np.logical_and(*condition)]=0
    mask[np.logical_not(np.logical_and(*condition))] = 255
    np.random.seed(0)
    im = np.random.randn(*mask.shape)[mask>th]

    totensor = transforms.ToTensor()
    mask = totensor(mask).cuda()
    applied_patch = torch.from_numpy(im).type(torch.cuda.FloatTensor).cuda()
    applied_patch.requires_grad=True
    mask = mask>th/255 #indices of the input image where mask is applied

        
    return applied_patch, mask

# Test the patch on dataset
def test_patch(patch_type, target, patch, test_loader, model):
    model.eval()
    test_total, test_actual_total, test_success = 0, 0, 0
    for (image, label) in test_loader:
        test_total += label.shape[0]
        # assert image.shape[0] == 1, 'Only one picture should be loaded each time.'
        image = image.cuda()
        label = label.cuda()
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        if predicted[0] != label and predicted[0].data.cpu().numpy() != target:
            test_actual_total += 1
            applied_patch, mask, x_location, y_location = mask_generation(patch_type, patch, image_size=(3, 224, 224))
            applied_patch = torch.from_numpy(applied_patch)
            mask = torch.from_numpy(mask)
            perturbed_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
            perturbed_image = perturbed_image.cuda()
            output = model(perturbed_image)
            _, predicted = torch.max(output.data, 1)
            if predicted[0].data.cpu().numpy() == target:
                test_success += 1
    return test_success / (test_actual_total+1)

normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
def patch_attack(args, image, applied_patch, mask, label, model, loss):
    model.eval()
    
    perturbed_image = image.type(torch.cuda.FloatTensor).cuda()
    perturbed_image[:,mask] = applied_patch

    per_image = normalize(perturbed_image)
    # Optimize the patch
    
    output = model(per_image)
    celoss = loss(output,label.repeat(output.shape[0]))
    patch_grad = torch.autograd.grad(celoss, applied_patch, retain_graph=False, create_graph=False)[0]

    applied_patch = args.lr * patch_grad.sign() + applied_patch
    applied_patch = torch.clamp(applied_patch, min=0, max=1)
    perturbed_image[:,mask] = applied_patch


    #to visualize the patch updates. not the same rgb
    _patch = applied_patch.detach().reshape((3,-1)).permute(1, 0).cpu().numpy()
    _s,_ = _patch.shape
    h = int(np.sqrt(_s))+1
    w = h
    patch_display = np.zeros((h*w,3))
    patch_display[:_s] = _patch
    patch_display = patch_display.reshape((h,w,3))
    patch_display = Image.fromarray((patch_display*255).astype(np.uint8))
    
    return perturbed_image, applied_patch, patch_display