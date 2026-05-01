import numpy as np
import torch


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
                raise ValueError("Attention head fusion type Not supported")

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

    max_value = np.max(mask)
    if max_value != 0:
        mask = mask / max_value

    return mask

def build_step_transition(prev_step_rollout_matrix, patch_attendance="identity"):
    """
    Build the transition matrix C_t from input tokens of step t
    to input tokens of step t-1.

    - current input CLS at step t is previous step output CLS
      => row 0 must be previous step CLS rollout row

    Patch_attendance:
        identity -> input patches are treated as corresponding
        zero     -> patch tokens are treated as new independent inputs
    """
    n_tokens = prev_step_rollout_matrix.size(-1)

    if patch_attendance == "identity":
        transition = torch.eye(n_tokens)
    elif patch_attendance == "zero":
        transition = torch.zeros(n_tokens, n_tokens)
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
    """

    def __init__(
        self,
        model,
        head_fusion="mean",
        discard_ratio=0.9,
        repeats=1,
        patch_attendance="identity",
        use_different_inputs=False,
    ):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio

        self.repeats = repeats
        self.patch_attendance = patch_attendance
        self.use_different_inputs = use_different_inputs

        self.attentions = []

        # Local rollout per recurrent step:
        # R_t: output(step t) -> input(step t)
        self.step_rollout_matrices = []
        self.step_rollout_masks = []

        # New merged implementation:
        # final output -> input(step t)
        self.final_to_step_input_matrices = []
        self.final_to_step_input_masks = []

        self.current_step_attentions = []
        self.per_step_logits = []

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
        """
        New recurrence aggregation.

        Computes rollout from the FINAL recurrent output to every recurrent input step.

        If repeats = 3, stores:
            final_to_step_input_masks[0]:
                output(step 3) -> input(step 1)

            final_to_step_input_masks[1]:
                output(step 3) -> input(step 2)

            final_to_step_input_masks[2]:
                output(step 3) -> input(step 3)

        List order follows step order:
            index 0 = input step 1
            index 1 = input step 2
            ...
        """
        self.final_to_step_input_matrices = []
        self.final_to_step_input_masks = []

        last_step = self.repeats - 1

        # Start with R_T:
        # final output -> input of final step
        final_to_current_input = self.step_rollout_matrices[last_step]

        matrices_final_to_current_input = [final_to_current_input]
        masks_final_to_current_input = [rollout_matrix_to_mask(final_to_current_input)]

        # Move backward:
        # output(final step) -> input(step T-1), ..., input(step 1)
        for step in range(last_step, 0, -1):
            # C_step:
            # input(step) -> input(step-1)
            current_input_to_prev_input = build_step_transition(
                self.step_rollout_matrices[step - 1],
                patch_attendance=self.patch_attendance,
            )

            # Compose:
            # final output -> input(step-1)
            final_to_current_input = torch.matmul(
                final_to_current_input,
                current_input_to_prev_input,
            )

            matrices_final_to_current_input.append(final_to_current_input)
            masks_final_to_current_input.append(rollout_matrix_to_mask(final_to_current_input))

        # Reverse so index corresponds to input step
        self.final_to_step_input_matrices = list(reversed(matrices_final_to_current_input))
        self.final_to_step_input_masks = list(reversed(masks_final_to_current_input))

        # final recurrent output -> input step 1
        return self.final_to_step_input_masks[0]

    def __call__(self, input_tensor):
        """
        Returns:
            final_rollout_mask (final recurrent output -> input step 1)

            final_to_step_input_masks:
                list of masks:
                final recurrent output -> input step t

            per_step_logits:
                classification logits from each recurrent step
        """
        self.attentions = []
        self.step_rollout_matrices = []
        self.step_rollout_masks = []
        self.final_to_step_input_matrices = []
        self.final_to_step_input_masks = []
        self.per_step_logits = []

        with torch.no_grad():
            # Recurrence starts from the learned cls_token
            print("1. cls_token.shape = ", self.model.cls_token.shape)
            cls_token = self.model.cls_token.expand(input_tensor.shape[0], -1, -1)
            print("2. cls_token.shape = ", cls_token.shape)
            print(cls_token)

            # if self.use_different_inputs and input_tensor.size() != self.repeats:
            #         print("Number of input tensors ({}) does not match number of recurrence steps ({})".format(input_tensor.size(), self.repeats))

            # Run recurrent steps one by one
            for step in range(self.repeats):
                self.current_step_attentions = []

                # model(x, cls_tok) -> (logits, new_cls_tok)
                # step_input = input_tensor[step] if self.use_different_inputs else input_tensor
                # output_logits, cls_token = self.model(step_input, cls_token)
                output_logits, cls_token = self.model(input_tensor, cls_token)

                # Store logits for this recurrent step
                self.per_step_logits.append(output_logits.detach().cpu())

                # Store attentions for this recurrent step
                self.attentions.append(self.current_step_attentions)

                # Local rollout:
                # output(step t) -> input(step t)
                step_rollout_matrix = compute_step_rollout_matrix(
                    self.current_step_attentions,
                    self.discard_ratio,
                    self.head_fusion
                )
                self.step_rollout_matrices.append(step_rollout_matrix)
                self.step_rollout_masks.append(rollout_matrix_to_mask(step_rollout_matrix))

        return self.rec_rollout(), self.final_to_step_input_masks, self.per_step_logits