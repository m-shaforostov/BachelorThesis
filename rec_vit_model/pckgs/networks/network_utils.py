import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))))
from config.definitions import NETWORKS_DIR
from pckgs.networks.recurrent_vision_transformer import RecurrentVisionTransformer


def load_trained_network(name, dataset, n_loops, run_no=0, pretrained=False, device=None, cuda_idx=0, method2=False,
                         reg_1000=False, on_off=False, tiny_patch=16, suffix='', sub_folder=''):
    if not device:
        device = f'cuda:{cuda_idx}' if torch.cuda.is_available() else "cpu"
    model_params = {
        'extra_tiny': (48, 6, 72, 6, 6),
        'tiny': (224, tiny_patch, 192, 12, 3),
    }
    ds_info = {
        'CIFAR_10': (10, 'cif10'),
        'CIFAR_100': (100, 'cif100'),
        'PET': (37, 'pet')
    }
    im_size, patch_size, embed_dim, depth, num_heads = model_params[name]
    model = RecurrentVisionTransformer(img_size=im_size,
                                       patch_size=patch_size,
                                       in_chans=3,
                                       num_classes=ds_info[dataset][0],
                                       embed_dim=embed_dim,
                                       depth=depth,
                                       num_heads=num_heads)

    if name == 'extra_tiny':
        model_path = NETWORKS_DIR + f'/{dataset}/{sub_folder}/{ds_info[dataset][1]}_{name}_k_{1}_run_{run_no}.pth'
    elif name == 'tiny':
        p = '_pretrained' if pretrained else ''
        m2 = '_method2' if method2 else ''
        r1000 = '_reg_1000' if reg_1000 else ''
        on_off = '_on_off' if on_off else ''
        model_path = NETWORKS_DIR + f'/{dataset}/{sub_folder}/{ds_info[dataset][1]}_{name}{p}_k_{1}_run_{run_no}' \
                                    f'{m2}{r1000}{on_off}{suffix}.pth'
    else:
        raise ValueError(f'Model {name} not recognized!')
    model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    model.to(device)
    print(f"Model loaded on {device}")

    model.eval()
    print('Model on eval mode')

    return model
