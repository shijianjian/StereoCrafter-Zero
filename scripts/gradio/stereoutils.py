from typing import Optional

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False
from lvdm.common import (
    checkpoint,
    exists,
    default,
)


class BNAttention():
    def __init__(self, start_step=0, end_step=50, total_steps=50, direction='uni'):
        self.total_steps = total_steps
        self.start_step = start_step
        self.end_step = end_step
        self.cur_step = 0
        self.cur_att_layer = 0
        self.direction = direction

    @torch.no_grad()
    def __call__(self, q: torch.Tensor, k, v, is_temporal: bool, split_dim=0):
        self.cur_att_layer += 1
        self.cur_step = self.cur_att_layer // (32 * 4)
        if (
            # is_temporal or
            (self.cur_step < self.start_step) or
            (self.cur_step >= self.end_step) or
            self.direction == "none"
        ):
            return q, k, v

        if self.direction == "uni":
            if split_dim == 0:
                q[k.size(0) // 2:] = q[:k.size(0) // 2]
                # k[k.size(0) // 2:] = k[:k.size(0) // 2]
                # v[v.size(0) // 2:] = v[:v.size(0) // 2]
            elif split_dim == 1:
                # k[:, k.size(1) // 2:] = k[:, :k.size(1) // 2]
                # v[:, v.size(1) // 2:] = v[:, :v.size(1) // 2]
                q[:, k.size(0) // 2:] = q[:, :k.size(0) // 2]
            else:
                raise RuntimeError
                
            return q, k, v
        if self.direction == "bi":
            if split_dim == 0:
                return (
                    q,
                    torch.cat([k[k.size(0) // 2:], k[:k.size(0) // 2]], dim=0),
                    torch.cat([v[v.size(0) // 2:], v[:v.size(0) // 2]], dim=0)
                )
            elif split_dim == 1:
                return (
                    q,
                    torch.cat([k[:, k.size(0) // 2:], k[:, :k.size(0) // 2]], dim=1),
                    torch.cat([v[:, v.size(0) // 2:], v[:, :v.size(0) // 2]], dim=1)
                )
            else:
                raise RuntimeError
        else:
            raise ValueError


def regiter_attention_editor_diffusers(model, editor):
    def ca_forward(self):
        def forward(x, context=None, mask=None):
            spatial_self_attn = (context is None)
            k_ip, v_ip, out_ip = None, None, None

            h = self.heads
            q = self.to_q(x)
            context = default(context, x)

            if self.image_cross_attention and not spatial_self_attn:
                context, context_image = context[:,:self.text_context_len,:], context[:,self.text_context_len:,:]
                k = self.to_k(context)
                v = self.to_v(context)
                k_ip = self.to_k_ip(context_image)
                v_ip = self.to_v_ip(context_image)
            else:
                if not spatial_self_attn:
                    context = context[:,:self.text_context_len,:]
                k = self.to_k(context)
                v = self.to_v(context)
            # n_q = q.shape[0]
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
            # assert False, (q.shape, k.shape, v.shape, h, n_q)
            # Update attention
            q, k, v = editor(q, k, v, is_temporal=self.is_temporal, split_dim=0)

            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
            if self.relative_position:
                len_q, len_k, len_v = q.shape[1], k.shape[1], v.shape[1]
                k2 = self.relative_position_k(len_q, len_k)
                sim2 = einsum('b t d, t s d -> b t s', q, k2) * self.scale # TODO check 
                sim += sim2
            del k

            if exists(mask):
                ## feasible for causal attention mask only
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b i j -> (b h) i j', h=h)
                sim.masked_fill_(~(mask > 0.5), max_neg_value)

            # attention, what we cannot get enough of
            sim = sim.softmax(dim=-1)

            out = torch.einsum('b i j, b j d -> b i d', sim, v)
            if self.relative_position:
                v2 = self.relative_position_v(len_q, len_v)
                out2 = einsum('b t s, t s d -> b t d', sim, v2) # TODO check
                out += out2
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

            ## for image cross-attention
            if k_ip is not None:
                k_ip, v_ip = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (k_ip, v_ip))
                # Update attention
                # _, k_ip, v_ip = editor(q, k_ip, v_ip, is_temporal=self.is_temporal, split_dim=0)
                sim_ip =  torch.einsum('b i d, b j d -> b i j', q, k_ip) * self.scale
                del k_ip
                sim_ip = sim_ip.softmax(dim=-1)
                out_ip = torch.einsum('b i j, b j d -> b i d', sim_ip, v_ip)
                out_ip = rearrange(out_ip, '(b h) n d -> b n (h d)', h=h)

            if out_ip is not None:
                if self.image_cross_attention_scale_learnable:
                    out = out + self.image_cross_attention_scale * out_ip * (torch.tanh(self.alpha)+1)
                else:
                    out = out + self.image_cross_attention_scale * out_ip
            
            return self.to_out(out)

        return forward
    
    def efficient_ca_forward(self):

        def efficient_forward(x, context=None, mask=None):
            spatial_self_attn = (context is None)
            k_ip, v_ip, out_ip = None, None, None

            q = self.to_q(x)
            context = default(context, x)

            if self.image_cross_attention and not spatial_self_attn:
                context, context_image = context[:,:self.text_context_len,:], context[:,self.text_context_len:,:]
                k = self.to_k(context)
                v = self.to_v(context)
                k_ip = self.to_k_ip(context_image)
                v_ip = self.to_v_ip(context_image)
            else:
                if not spatial_self_attn:
                    context = context[:,:self.text_context_len,:]
                k = self.to_k(context)
                v = self.to_v(context)

            b, _, _ = q.shape
            q, k, v = map(
                lambda t: t.unsqueeze(3)
                .reshape(b, t.shape[1], self.heads, self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b * self.heads, t.shape[1], self.dim_head)
                .contiguous(),
                (q, k, v),
            )
            # Update attention
            q, k, v = editor(q, k, v, is_temporal=self.is_temporal, split_dim=0)
            # actually compute the attention, what we cannot get enough of
            out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=None)

            ## for image cross-attention
            if k_ip is not None:
                k_ip, v_ip = map(
                    lambda t: t.unsqueeze(3)
                    .reshape(b, t.shape[1], self.heads, self.dim_head)
                    .permute(0, 2, 1, 3)
                    .reshape(b * self.heads, t.shape[1], self.dim_head)
                    .contiguous(),
                    (k_ip, v_ip),
                )
                # _, k_ip, v_ip = editor(q, k_ip, v_ip, is_temporal=self.is_temporal, split_dim=0)
                out_ip = xformers.ops.memory_efficient_attention(q, k_ip, v_ip, attn_bias=None, op=None)
                out_ip = (
                    out_ip.unsqueeze(0)
                    .reshape(b, self.heads, out.shape[1], self.dim_head)
                    .permute(0, 2, 1, 3)
                    .reshape(b, out.shape[1], self.heads * self.dim_head)
                )

            if exists(mask):
                raise NotImplementedError
            out = (
                out.unsqueeze(0)
                .reshape(b, self.heads, out.shape[1], self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b, out.shape[1], self.heads * self.dim_head)
            )
            if out_ip is not None:
                if self.image_cross_attention_scale_learnable:
                    out = out + self.image_cross_attention_scale * out_ip * (torch.tanh(self.alpha)+1)
                else:
                    out = out + self.image_cross_attention_scale * out_ip
            
            return self.to_out(out)

        return efficient_forward

    def register_editor(net, count):
        for name, subnet in net.named_children():
            # if net.__class__.__name__ == 'Attention':  # spatial Transformer layer
            if 'CrossAttention' in net.__class__.__name__:
                if XFORMERS_IS_AVAILBLE and net.temporal_length is None:
                    net.forward = efficient_ca_forward(net)
                else:
                    net.forward = ca_forward(net)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count)
        return count

    cross_att_count = 0
    try:
        sub_nets = model.model.diffusion_model.named_children()
    except: 
        sub_nets = model.unet.named_children()
    for net_name, net in sub_nets:
        if "down" in net_name or "input" in net_name:
            cross_att_count += register_editor(net, 0)
        elif "mid" in net_name:
            cross_att_count += register_editor(net, 0)
        elif "up" in net_name or "output" in net_name:
            cross_att_count += register_editor(net, 0)
    editor.num_att_layers = cross_att_count
