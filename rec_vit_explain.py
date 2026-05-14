import os
import argparse

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from rec_vit_rollout import RecVITAttentionRollout
from rec_vit_model.pckgs.networks.network_utils import load_trained_network
from attention_map_evaluation import evaluate_attention_maps
from reveal_segmentation import make_trimap


CLASSES = {
    "CIFAR_10": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
    "PET": [
        "Abyssinian", "American Bulldog", "American Pit Bull Terrier", "Basset Hound", "Beagle", "Bengal", "Birman",
        "Bombay", "Boxer", "British Shorthair", "Chihuahua", "Egyptian Mau", "English Cocker Spaniel",
        "English Setter", "German Shorthaired", "Great Pyrenees", "Havanese", "Japanese Chin", "Keeshond",
        "Leonberger", "Maine Coon", "Miniature Pinscher", "Newfoundland", "Persian", "Pomeranian", "Pug",
        "Ragdoll", "Russian Blue", "Saint Bernard", "Samoyed", "Scottish Terrier", "Shiba Inu", "Siamese",
        "Sphynx", "Staffordshire Bull Terrier", "Wheaten Terrier", "Yorkshire Terrier"
    ],
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image_path', type=str,
                        help='Input image path')
    parser.add_argument('--output_dir', type=str, default='./rollout_results/',
                        help='Output directory')
    parser.add_argument('--head_fusion', type=str, default='max',
                        help='How to fuse the attention heads for attention rollout. Can be mean/max/min')
    parser.add_argument('--discard_ratio', type=float, default=0.9,
                        help='How many of the lowest attention paths should be discarded')

    parser.add_argument('--model_name', type=str, default='tiny',
                        help='RecViT model size: tiny or extra_tiny')
    parser.add_argument('--dataset', type=str, default='CIFAR_10',
                        help='Dataset used for the trained RecViT checkpoint: CIFAR_10, CIFAR_100, or PET')
    parser.add_argument('--n_loops', type=int, default=1,
                        help='Number of recurrent loops used by the RecViT model')
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
    parser.add_argument('--tiny_patch', type=int, default=16,
                        help='Patch size for tiny model, use 8 for PET pretrained checkpoints if needed')

    parser.add_argument('--trimap_path', type=str,
                        help='Path to raw PET trimap-like image used to generate segmentation mask')

    parser.add_argument('--patch_attendance', type=str, default='identity',
                        help="How to connect patch tokens across recurrent steps: 'identity' or 'zero'")
    parser.add_argument('--use_different_inputs', action='store_true', default=False,
                        help="Do you provide different inputs across recurrent steps?")

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


def get_predictions(logits, dataset):
    predictions = []
    for i in logits:
        predictions.append(CLASSES[dataset][int(i.argmax(dim=1))])
    return predictions


def get_attention_overlay(img, mask):
    np_img = np.array(img)[:, :, ::-1]  # RGB -> BGR
    resized_mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
    overlay = show_mask_on_image(np_img, resized_mask)
    return overlay


def save_attention_overlay(img, mask, output_path):
    overlay = get_attention_overlay(img, mask)
    cv2.imwrite(output_path, overlay)


def save_trimap_attention_overlay(img, trimap, attention_mask, output_path):
    # Get attention overlay (already colored heatmap on image)
    attention_overlay_bgr = get_attention_overlay(img, attention_mask)
    attention_overlay_rgb = cv2.cvtColor(attention_overlay_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    base = np.array(img).astype(np.float32)

    # Resize trimap
    trimap_resized = cv2.resize(
        trimap,
        (base.shape[1], base.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    # Create BORDER-only mask
    border_mask = (trimap_resized == 128)

    # Create overlay with ONLY border
    overlay = attention_overlay_rgb.copy()

    # Highlight border in a strong color (yellow)
    overlay[border_mask] = [255, 255, 0]

    # Blend with attention map
    result = cv2.addWeighted(attention_overlay_rgb, 0.75, overlay, 0.25, 0)

    result = np.clip(result, 0, 255).astype(np.uint8)

    cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    args = get_args()
    device = 'cuda:0' if args.use_cuda else 'cpu'

    print("PARAMS:\n"
          "name={}\n"
          "dataset={}\n"
          "n_loops={}\n"
          "run_no={}\n"
          "pretrained={}\n"
          "device={}\n"
          "method2={}\n"
          "reg_1000={}\n"
          "on_off={}\n"
          "patch_attendance={}\n"
          "use_different_inputs={}\n".format(
        args.model_name,
        args.dataset,
        args.n_loops,
        args.run_no,
        args.pretrained,
        device,
        args.method2,
        args.reg_1000,
        args.on_off,
        args.patch_attendance,
        args.use_different_inputs))

    model = load_trained_network(
        name=args.model_name,
        dataset=args.dataset,
        n_loops=args.n_loops,
        run_no=args.run_no,
        pretrained=args.pretrained,
        device=device,
        method2=args.method2,
        reg_1000=args.reg_1000,
        on_off=args.on_off,
        tiny_patch=args.tiny_patch,
    )

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    img = Image.open(args.image_path).convert('RGB')
    img = img.resize((224, 224))
    input_tensor = transform(img).unsqueeze(0)

    if args.use_cuda:
        input_tensor = input_tensor.cuda()

    print("Doing RecViT Attention Rollout")
    attention_rollout = RecVITAttentionRollout(
        model,
        head_fusion=args.head_fusion,
        discard_ratio=args.discard_ratio,
        repeats=args.n_loops,
        patch_attendance=args.patch_attendance,
        use_different_inputs=args.use_different_inputs,
    )

    final_rollout_mask, final_to_step_input_masks, per_step_logits = attention_rollout(input_tensor)

    print(get_predictions(per_step_logits, args.dataset))

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset == "PET":
        attention_maps = {
            "final_rollout": final_rollout_mask,
        }

        for step, step_mask in enumerate(final_to_step_input_masks):
            attention_maps[f"final_to_input_step_{step + 1}"] = step_mask

        if args.trimap_path is not None:
            eval_output_dir = os.path.join(args.output_dir, "trimap")
            os.makedirs(eval_output_dir, exist_ok=True)

            trimap = make_trimap(args.trimap_path)

            evaluate_attention_maps(
                trimap=trimap,
                attention_maps=attention_maps,
                output_dir=eval_output_dir,
            )

            for map_name, attention_mask in attention_maps.items():
                overlay_path = os.path.join(eval_output_dir, f"{map_name}_attention_trimap_overlay.png")
                save_trimap_attention_overlay(
                    img=img,
                    trimap=trimap,
                    attention_mask=attention_mask,
                    output_path=overlay_path,
                )

    np_img = np.array(img)[:, :, ::-1]
    cv2.imwrite(os.path.join(args.output_dir, "input.png"), np_img)

    name = "rec_rollout_{}_loops_{}_{:.3f}_{}.png".format(
        args.n_loops,
        args.patch_attendance,
        args.discard_ratio,
        args.head_fusion
    ).replace(" ", "")

    path = os.path.join(args.output_dir, name)
    save_attention_overlay(img, final_rollout_mask, path)

    for step, step_mask in enumerate(final_to_step_input_masks):
        name = "rec_rollout_final_to_input_step_{}_{}_loops_{}_{:.3f}_{}.png".format(
            step + 1,
            args.n_loops,
            args.patch_attendance,
            args.discard_ratio,
            args.head_fusion
        ).replace(" ", "")

        path = os.path.join(args.output_dir, name)
        save_attention_overlay(img, step_mask, path)