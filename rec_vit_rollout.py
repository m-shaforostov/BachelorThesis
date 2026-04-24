import numpy as np

import torch
from PIL import Image
import sys
from torchvision import transforms
import cv2


def compute_step_rollout_matrix(attentions, discard_ratio, head_fusion):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)

    return result


def rollout_matrix_to_mask(result):
    mask = result[0, 0, 1:]
    width = int(mask.size(-1) ** 0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask

def build_step_transition(prev_step_rollout_matrix, patch_attendance="identity"):
    """
    Build the transition matrix C_t from input tokens of step t
    to input tokens of step t-1.

    - current input CLS at step t is previous step output CLS
      => row 0 must be previous step CLS rollout row

    - patch_attendance == "identity":
        use when the same image patches are reused across recurrent steps
    - patch_attendance == "zero":
        use when patches at each step are treated as new roots
    """
    if patch_attendance == "identity":
        # Same-patch assumption: patch i at step t corresponds to patch i at step t-1
        n_tokens = prev_step_rollout_matrix.size(-1)
        transition = torch.eye(n_tokens)
    elif patch_attendance == "zero":
        # Different-patch-root assumption: no direct patch-to-patch connection across steps
        transition = torch.zeros_like(prev_step_rollout_matrix)
    else:
        raise ValueError(
            f"patch_attendance='{patch_attendance}' not supported. "
            f"Use 'identity' or 'zero'."
        )
    # Current input CLS at step t = previous step output CLS
    transition[0, :] = prev_step_rollout_matrix[0, 0, :]

    return transition


class RecVITAttentionRollout:
    """
    Attention Rollout adapted to RecViT.

    Confirmed implementation logic:
    - runs recurrent steps one by one
    - stores attentions separately for each step: attentions[step][layer]
    - computes standard rollout independently for each step
    - derives overall rollout across recurrent steps through explicit step transitions

    Assumption:
    - final output of interest is the LAST recurrent step
    - patch_attendance decides how to connect patch inputs across steps:
        * identity -> same patch tokens reused
        * zero     -> different patch roots per step
    """
    def __init__(self, model, head_fusion="mean", discard_ratio=0.9, repeats=1, patch_attendance="identity", use_different_inputs=False):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.attentions = []

        self.repeats = repeats
        self.patch_attendance = patch_attendance
        self.use_different_inputs = use_different_inputs

        # Standard rollout per recurrent step
        self.step_rollout_matrices = []
        self.step_rollout_masks = []

        # Overall rollout from each step output back to step-1 input tokens
        # self.overall_rollout_matrices = []
        self.overall_rollout_masks = []

        # Temporary step buffer used by hooks
        self.current_step_attentions = []

        for name, module in model.named_modules():
            if "attn" in name and hasattr(module, "qkv"):
                print("Hooking:", name)
                module.register_forward_hook(self.get_attention)

    def get_attention(self, module, input, output):
        x = input[0]

        B, N, C = x.shape
        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * module.scale
        attn = attn.softmax(dim=-1)

        # save attentions grouped by steps
        self.current_step_attentions.append(attn.detach().cpu())

    def rec_rollout(self):
        # Derive overall rollout across the recurrent network
        # B_t maps input tokens of step t to input tokens of step 1
        # Start with B_1 = I
        n_tokens = self.step_rollout_matrices[0].size(-1)
        step_input_to_first_input = torch.eye(n_tokens).unsqueeze(0)

        for step in range(self.repeats):
            if step > 0:
                # C_t: input(step t) -> input(step t-1)
                curr_step_input_to_prev_input = build_step_transition(
                    self.step_rollout_matrices[step - 1],
                    patch_attendance=self.patch_attendance
                )
                # B_t: input(step t) -> input(step 1)
                # B_t = C_t * B_(t-1)
                step_input_to_first_input = torch.matmul(curr_step_input_to_prev_input,
                                                              step_input_to_first_input)
            # G_t: last_layer(step t) -> input(step 1)
            # G_t = R_t * B_t
            overall_rollout_matrix_curr_step = torch.matmul(
                self.step_rollout_matrices[step],
                step_input_to_first_input
            )

            # self.overall_rollout_matrices.append(overall_rollout_matrix_curr_step)
            self.overall_rollout_masks.append(rollout_matrix_to_mask(overall_rollout_matrix_curr_step))

        return self.overall_rollout_masks[-1]

    def __call__(self, input_tensor):
        """
        Returns the final overall rollout mask from the LAST recurrent step.

        Also stores:
        - self.attentions[step][layer]
        - self.step_rollout_matrices[step]
        - self.step_rollout_masks[step]
        - self.overall_rollout_matrices[step]
        - self.overall_rollout_masks[step]
        """
        self.attentions = []
        self.step_rollout_matrices = []
        self.step_rollout_masks = []
        # self.overall_rollout_matrices = []
        self.overall_rollout_masks = []

        with torch.no_grad():
            # Recurrence starts from the learned cls_token
            print("1. cls_token.shape = ", self.model.cls_token.shape)
            cls_token = self.model.cls_token.expand(input_tensor.shape[0], -1, -1)
            print("2. cls_token.shape = ", cls_token.shape)

            if self.use_different_inputs and input_tensor.size() != self.repeats:
                    print("Number of input tensors ({}) does not match number of recurrence steps ({})".format(input_tensor.size(), self.repeats))

            # Run recurrent steps one by one
            for step in range(self.repeats):
                self.current_step_attentions = []

                # model(x, cls_tok) -> (logits, new_cls_tok)
                step_input = input_tensor[step] if self.use_different_inputs else input_tensor
                _, cls_token = self.model(step_input, cls_token)

                # Store attentions for this recurrent step
                self.attentions.append(self.current_step_attentions)

                # Standard rollout inside this step only
                step_rollout_matrix = compute_step_rollout_matrix(
                    self.current_step_attentions,
                    self.discard_ratio,
                    self.head_fusion
                )
                self.step_rollout_matrices.append(step_rollout_matrix)
                self.step_rollout_masks.append(rollout_matrix_to_mask(step_rollout_matrix))

        return self.rec_rollout(), self.step_rollout_masks