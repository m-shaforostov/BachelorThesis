import numpy as np

import argparse
import sys
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import cv2

from vit_rollout import VITAttentionRollout
from vit_grad_rollout import VITAttentionGradRollout
from rec_vit_model.pckgs.networks.network_utils import load_trained_network


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image_path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--head_fusion', type=str, default='max',
                        help='How to fuse the attention heads for attention rollout. \
                        Can be mean/max/min')
    parser.add_argument('--discard_ratio', type=float, default=0.9,
                        help='How many of the lowest 14x14 attention paths should we discard')
    parser.add_argument('--category_index', type=int, default=None,
                        help='The category index for gradient rollout')

    # For load_trained_network(...)
    parser.add_argument('--model_name', type=str, default='tiny',
                        help='RecViT model size: tiny or extra_tiny')
    parser.add_argument('--dataset', type=str, default='CIFAR_10',
                        help='Dataset used for the trained RecViT checkpoint: CIFAR_10, CIFAR_100, or PET')
    parser.add_argument('--n_loops', type=int, default=1,
                        help='Number of recurrent loops used when the checkpoint was trained')
    parser.add_argument('--run_no', type=int, default=0,
                        help='Checkpoint run number')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='Load the checkpoint variant trained from pretrained ViT initialization')
    parser.add_argument('--method2', action='store_true', default=False,
                        help='Use method2 checkpoint naming')
    parser.add_argument('--reg_1000', action='store_true', default=False,
                        help='Use reg_1000 checkpoint naming')
    parser.add_argument('--on_off', action='store_true', default=False,
                        help='Use on_off checkpoint naming')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU")
    else:
        print("Using CPU")

    return args


def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


class RecViTForExplainability(nn.Module):
    """
    Adapter for the existing rollout code.

    Existing rollout code expects:
        model(input_tensor) -> logits

    RecurrentVisionTransformer provides:
        model(x, cls_tok) -> (logits, new_tokens)

    So this wrapper:
    - creates the initial class token automatically
    - returns only logits
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        cls_tok = self.model.cls_token.expand(x.shape[0], -1, -1)
        logits, _ = self.model(x, cls_tok)
        return logits


if __name__ == '__main__':
    args = get_args()
    device = 'cuda:0' if args.use_cuda else 'cpu'

    # Load the trained RecViT checkpoint using the project utility loader.
    base_model = load_trained_network(
        name=args.model_name,
        dataset=args.dataset,
        n_loops=args.n_loops,
        run_no=args.run_no,
        pretrained=args.pretrained,
        device=device,
        method2=args.method2,
        reg_1000=args.reg_1000,
        on_off=args.on_off,
    )

    # Wrap RecViT so Rollout can keep using model(input_tensor).
    model = RecViTForExplainability(base_model)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    img = Image.open(args.image_path)
    img = img.resize((224, 224))
    input_tensor = transform(img).unsqueeze(0)

    if args.use_cuda:
        input_tensor = input_tensor.cuda()

    if args.category_index is None:
        print("Doing Attention Rollout")
        attention_rollout = VITAttentionRollout(model, head_fusion=args.head_fusion,
            discard_ratio=args.discard_ratio)
        mask = attention_rollout(input_tensor)
        name = "attention_rollout_{:.3f}_{}.png".format(args.discard_ratio, args.head_fusion)
    else:
        print("Doing Gradient Attention Rollout")
        grad_rollout = VITAttentionGradRollout(model, discard_ratio=args.discard_ratio)
        mask = grad_rollout(input_tensor, args.category_index)
        name = "grad_rollout_{}_{:.3f}_{}.png".format(args.category_index,
            args.discard_ratio, args.head_fusion)

    np_img = np.array(img)[:, :, ::-1]
    mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
    mask = show_mask_on_image(np_img, mask)
    cv2.imshow("Input Image", np_img)
    cv2.imshow(name, mask)
    cv2.imwrite("input.png", np_img)
    cv2.imwrite(name, mask)
    cv2.waitKey(-1)