# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.optim import AdamW

class AdamWFP32Copy(AdamW):
    r"""Implements AdamW algorithm with FP32 copy for parameters.

    This implementation keeps a float32 copy of parameters to perform optimizer calculations
    in higher precision, helping with training stability.
    """

    @torch.no_grad()
    def step(self, closure=None, scale=1.0):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            scale (float, optional): Scale factor for gradients (for mixed precision training)
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Extract parameters from group
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            amsgrad = group['amsgrad']
            maximize = group.get('maximize', False)

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # Get gradient and apply scaling if necessary
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamWFP32Copy does not support sparse gradients")
                
                # Scale the gradient for mixed precision training
                grad = grad.float() / scale

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Create a float32 copy of the parameter
                    state["float32copy"] = p.to(torch.float32, memory_format=torch.preserve_format)
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        state["float32copy"], memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        state["float32copy"], memory_format=torch.preserve_format
                    )
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            state["float32copy"], memory_format=torch.preserve_format
                        )

                # Get FP32 parameter and state variables
                fp32_p = state["float32copy"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                step = state["step"]
                step += 1
                state["step"] = step

                if maximize:
                    grad = -grad

                # Perform weight decay
                if weight_decay != 0:
                    fp32_p.mul_(1 - lr * weight_decay)

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                step_size = lr / bias_correction1
                
                if amsgrad:
                    # Maintains the maximum of all exp moving avg of sq grad values
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    
                    # Use the max for normalizing
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # Update parameters
                fp32_p.addcdiv_(exp_avg, denom, value=-step_size)
                
                # Copy updated parameters back to original dtype and tensor
                p.copy_(fp32_p)

        return loss