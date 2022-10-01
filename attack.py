# Adversarial Patch Attack
# Created by Junbo Zhao 2020/3/17

"""
Reference:
[1] Tom B. Brown, Dandelion Mané, Aurko Roy, Martín Abadi, Justin Gilmer
    Adversarial Patch. arXiv:1712.09665
"""

from genericpath import exists
import torch
import torch.nn as nn

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.utils import save_image

import pathlib
import argparse
import csv
import os
import numpy as np


from patch_utils import*
from utils import*


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    # Load the model
    model = resnet50(weights=ResNet50_Weights.DEFAULT).eval().cuda()
    loss = nn.CrossEntropyLoss()   
    
    # Load the datasets
    train_loader, test_loader, train_total, test_total = dataloader(args, 50000)
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                        std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                    transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                        std = [ 1., 1., 1. ]),])
                 

    # Test the accuracy of model on trainset and testset
    testset_acc, labels, h, w = test(model, test_loader)
    print('Clean set acc: {:.3f}% / label: {}'.format(100*testset_acc, labels[0].item()))

    best_patch_epoch, best_patch_success_rate, test_acc = 0, 1, 1


    exp_dir = os.path.join('logs',args.experiment_name)
    path = pathlib.Path(exp_dir)
    path.mkdir(exist_ok=True)

    eval_per = 10
    applied_patch, mask = mask_generation(args, h, w)
    # Generate the patch
    for iter in range(0,args.max_iteration):
        if test_acc > args.probability_threshold and iter < args.max_iteration: 
            for (images, _) in train_loader:
                images = images.cuda()
                label = labels[0].cuda()
                perturbed_image, applied_patch, patch_display = patch_attack(args, images, applied_patch, mask, label, model, loss)

            #evaluate patch
            if iter % eval_per == 0:
                model.eval()
                all_preds = []

                with torch.no_grad():
                    for (images, _) in test_loader:
                        perturbed_image = images.type(torch.cuda.FloatTensor).cuda()
                        perturbed_image[:,mask] = applied_patch

                        perturbed_image = normalize(perturbed_image)
                        outputs = model(perturbed_image)

                        predicted = outputs.argmax(dim=1)
                        all_preds.append(predicted)

                all_preds = torch.concat(all_preds)
                test_acc = torch.sum(all_preds == labels[:test_total])/test_total
                print("Iteration:{} test set accuracy: {:.3f}%".format(iter, 100 * test_acc))
                
                patch_display.save(os.path.join(exp_dir,"patch_{}.png".format(str(iter))))
                save_image(invTrans(perturbed_image[:-1]), os.path.join(exp_dir,"imgs_{}.png".format(str(iter))))

                # Record the statistics
                with open(os.path.join(exp_dir,'log.csv'),'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([iter, test_acc.item()])

                if test_acc < best_patch_success_rate:
                    best_patch_success_rate = test_acc
                    best_patch_epoch = iter
                    # save_image(patch, os.path.join(exp_dir , "patch_best.png"))

                # Load the statistics and generate the line
                log_generation(exp_dir)

    print("The best patch is found at iter {} with success rate {:.2f}% on testset".format(best_patch_epoch, 100 * best_patch_success_rate))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=-1, help="batch size")
    parser.add_argument('--num_workers', type=int, default=2, help="num_workers")
    parser.add_argument('--noise_percentage', type=float, default=0.01, help="percentage of the patch size compared with the image size")
    parser.add_argument('--probability_threshold', type=float, default=0.05, help="minimum target probability")
    parser.add_argument('--lr', type=float, default=4/255, help="learning rate")
    parser.add_argument('--max_iteration', type=int, default=1000, help="max iteration")
    parser.add_argument('--data_dir', type=str, default='dataset/shark', help="dir of the dataset")
    parser.add_argument('--patch_type', type=str, default='rectangle', help="type of the patch")
    parser.add_argument('--GPU', type=str, default='0', help="index pf used GPU")
    parser.add_argument('--experiment_name', type=str, default='test_default', help='dir where you save logs')
    parser.add_argument('--resize', default=True, action='store_true', help='whether to resize')
    args = parser.parse_args()
    main(args)