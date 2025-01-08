from typing import Optional

import torch
from torch.nn.utils import parameters_to_vector
from torchviz import make_dot

import bitsandbytes.functional as F
from bitsandbytes.optim.optimizer import Optimizer8bit, MockArgs


def parameter_grad_to_vector(param):
    return param.grad.view(-1)


def parameter_to_vector(param):
    return param.view(-1)


def visualize_graph(tensor, **kwargs):
    dot = make_dot(tensor, **kwargs)
    dot.render("model_graph", format="png")  # 将图保存为 PNG 文件
    dot.view()  # 直接查看图


def compute_vjp(cur_p, prev_p):
    cur_grad = parameter_grad_to_vector(cur_p).requires_grad_(True)
    delta_p_flatten = parameters_to_vector(cur_p - prev_p)
    vjp_param = torch.autograd.grad(outputs=cur_grad,
                                    inputs=cur_p,
                                    grad_outputs=delta_p_flatten,
                                    retain_graph=True)
    return vjp_param


class LEAD_Adam(Optimizer8bit):
    def __init__(
            self,
            params,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            optim_bits=8,
            args=None,
            min_8bit_size=4096,
            percentile_clipping=100,
            block_wise=True,
            max_unorm=0.0,
            skip_zeros=False,
            is_paged=False,
            alpha=0.0,
            alpha_vjp=0.0,
            alpha_grad=1.0,
            t_alpha: Optional[int] = None,
            t_beta3: Optional[int] = None,
    ):
        """
        Base 2-state update optimizer class.

        Arguments:
            optimizer_name (`str`):
                The name of the optimizer.
            params (`torch.tensor`):
                The input parameters to optimize.
            lr (`float`, defaults to 1e-3):
                The learning rate.
            betas (`tuple`, defaults to (0.9, 0.999)):
                The beta values for the optimizer.
            eps (`float`, defaults to 1e-8):
                The epsilon value for the optimizer.
            weight_decay (`float`, defaults to 0.0):
                The weight decay value for the optimizer.
            optim_bits (`int`, defaults to 32):
                The number of bits of the optimizer state.
            args (`object`, defaults to `None`):
                An object with additional arguments.
            min_8bit_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 8-bit optimization.
            percentile_clipping (`int`, defaults to 100):
                Adapts clipping threshold automatically by tracking the last 100 gradient norms and clipping the gradient at a certain percentile to improve stability.
            block_wise (`bool`, defaults to `True`):
                Whether to independently quantize each block of tensors to reduce outlier effects and improve stability.
            max_unorm (`float`, defaults to 0.0):
                The maximum value to normalize each block with.
            skip_zeros (`bool`, defaults to `False`):
                Whether to skip zero values for sparse gradients and models to ensure correct updates.
            is_paged (`bool`, defaults to `False`):
                Whether the optimizer is a paged optimizer or not.
            alpha (`float`, defaults to 0.0):
                The alpha value for the AdEMAMix optimizer.
            t_alpha (`Optional[int]`, defaults to `None`):
                Number of iterations for alpha scheduling with AdEMAMix.
            t_beta3 (`Optional[int]`, defaults to `None`):
                Number of iterations for beta scheduling with AdEMAMix.

        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if isinstance(betas, str):
            # format: '(beta1, beta2)'
            betas = betas.replace("(", "").replace(")", "").strip().split(",")
            betas = [float(b) for b in betas]
        for i in range(len(betas)):
            if not 0.0 <= betas[i] < 1.0:
                raise ValueError(f"Invalid beta parameter at index {i}: {betas[i]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, alpha=alpha, t_alpha=t_alpha, t_beta3=t_beta3,
            alpha_vjp=alpha_vjp, alpha_grad=alpha_grad
        )

        super().__init__(params, defaults, optim_bits, is_paged)

        if args is None:
            args = {}
            args["optim_bits"] = optim_bits
            args["percentile_clipping"] = 100
            args["min_8bit_size"] = min_8bit_size
            args["percentile_clipping"] = percentile_clipping
            args["block_wise"] = block_wise
            args["max_unorm"] = max_unorm
            args["skip_zeros"] = skip_zeros

            self.args = MockArgs(args)
        else:
            self.args = args

        self.optimizer_name = 'adam'

    @torch.no_grad()
    def init_state(self, group, p, gindex, pindex):
        config = self.get_config(gindex, pindex, group)

        if config["optim_bits"] == 32:
            dtype = torch.float32
        elif config["optim_bits"] == 8:
            dtype = torch.uint8
        else:
            raise NotImplementedError(f'Amount of optimizer bits not supported: {config["optim_bits"]}')

        if p.numel() < config["min_8bit_size"]:
            dtype = torch.float32

        state = self.state[p]
        state["step"] = 0

        if dtype == torch.float32:
            state["state1"] = self.get_state_buffer(p, dtype=torch.float32)
            state["state2"] = self.get_state_buffer(p, dtype=torch.float32)
        elif dtype == torch.uint8:
            if state["step"] == 0:
                if "dynamic" not in self.name2qmap:
                    self.fill_qmap()
                self.name2qmap["dynamic"] = self.name2qmap["dynamic"].to(p.device)
                self.name2qmap["udynamic"] = self.name2qmap["udynamic"].to(p.device)

            state["state1"] = self.get_state_buffer(p, dtype=torch.uint8)
            state["qmap1"] = self.name2qmap["dynamic"]

            state["state2"] = self.get_state_buffer(p, dtype=torch.uint8)
            state["qmap2"] = self.name2qmap["udynamic"]

            if config["block_wise"]:
                n = p.numel()
                blocks = n // 256
                blocks += 1 if n % 256 > 0 else 0

                state["absmax1"] = torch.zeros((blocks,), dtype=torch.float32, device=p.device)
                state["absmax2"] = torch.zeros((blocks,), dtype=torch.float32, device=p.device)
            else:
                state["max1"] = torch.zeros((1,), dtype=torch.float32, device=p.device)
                state["new_max1"] = torch.zeros((1,), dtype=torch.float32, device=p.device)
                state["max2"] = torch.zeros((1,), dtype=torch.float32, device=p.device)
                state["new_max2"] = torch.zeros((1,), dtype=torch.float32, device=p.device)

        state['prev_params'] = p.clone()  # Store previous parameters for VJP

        if config["percentile_clipping"] < 100:
            state["gnorm_vec"] = torch.zeros((100,), device=p.device)

        if config["max_unorm"] > 0.0:
            state["unorm_vec"] = torch.zeros((1,), device=p.device)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Arguments:
            closure (`Callable`, *optional*, defaults to `None`):
                A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        overflows = []

        if not self.initialized:
            self.check_overrides()
            self.to_gpu()  # needed for fairseq pure fp16 training
            self.initialized = True

        # if self.is_paged: self.page_mng.prefetch_all()
        for gindex, group in enumerate(self.param_groups):
            for pindex, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    self.init_state(group, p, gindex, pindex)

                vjp_param = compute_vjp(p, state['prev_params'])

                state["prev_params"] = p.clone()
                state["vjp"] = vjp_param

                self.prefetch_state(p)
                self.update_step(group, p, gindex, pindex)
                torch.cuda.synchronize()
        if self.is_paged:
            # all paged operation are asynchronous, we need
            # to sync to make sure all tensors are in the right state
            torch.cuda.synchronize()

        return loss

    @torch.no_grad()
    def update_step(self, group, p, gindex, pindex):
        # avoid update error from non-contiguous memory layout
        p.data = p.data.contiguous()
        p.grad = p.grad.contiguous()

        state = self.state[p]
        grad = p.grad
        config = self.get_config(gindex, pindex, group)

        state["step"] += 1
        step = state["step"]

        # Computation of the gradient with respect to the VJP from parameters
        vjp = state["vjp"]
        alpha_grad = group["alpha_grad"]
        alpha_vjp = group["alpha_vjp"]
        grad = alpha_grad * grad - alpha_vjp * vjp

        if config["percentile_clipping"] < 100:
            current_gnorm, clip_value, gnorm_scale = F.percentile_clipping(
                grad,
                state["gnorm_vec"],
                step,
                config["percentile_clipping"],
            )
        else:
            gnorm_scale = 1.0

        if state["state1"].dtype == torch.float:
            print("Warning: using 32-bit optimizer for 8-bit parameter")
            F.optimizer_update_32bit(
                self.optimizer_name,
                grad,
                p,
                state["state1"],
                config["betas"][0],
                config["eps"],
                step,
                config["lr"],
                state["state2"],
                config["betas"][1],
                config["betas"][2] if len(config["betas"]) >= 3 else 0.0,
                config["alpha"],
                config["weight_decay"],
                gnorm_scale,
                state["unorm_vec"] if config["max_unorm"] > 0.0 else None,
                max_unorm=config["max_unorm"],
                skip_zeros=config["skip_zeros"],
            )
        elif state["state1"].dtype == torch.uint8 and not config["block_wise"]:
            print("Warning: using a deprecated 8-bit optimizer")
            F.optimizer_update_8bit(
                self.optimizer_name,
                grad,
                p,
                state["state1"],
                state["state2"],
                config["betas"][0],
                config["betas"][1],
                config["eps"],
                step,
                config["lr"],
                state["qmap1"],
                state["qmap2"],
                state["max1"],
                state["max2"],
                state["new_max1"],
                state["new_max2"],
                config["weight_decay"],
                gnorm_scale=gnorm_scale,
                unorm_vec=state["unorm_vec"] if config["max_unorm"] > 0.0 else None,
                max_unorm=config["max_unorm"],
            )

            # swap maxes
            state["max1"], state["new_max1"] = state["new_max1"], state["max1"]
            state["max2"], state["new_max2"] = state["new_max2"], state["max2"]
        elif state["state1"].dtype == torch.uint8 and config["block_wise"]:
            F.optimizer_update_8bit_blockwise(
                self.optimizer_name,
                grad,
                p,
                state["state1"],
                state["state2"],
                config["betas"][0],
                config["betas"][1],
                config["betas"][2] if len(config["betas"]) >= 3 else 0.0,
                config["alpha"],
                config["eps"],
                step,
                config["lr"],
                state["qmap1"],
                state["qmap2"],
                state["absmax1"],
                state["absmax2"],
                config["weight_decay"],
                gnorm_scale=gnorm_scale,
                skip_zeros=config["skip_zeros"],
            )
