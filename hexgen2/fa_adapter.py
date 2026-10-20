"""Dispatch FlashAttention to validated Turing or upstream CUDA kernels."""
import importlib
import torch

CALLS = {}
AUDIT = False
ERRORS = []
NAMES = ['flash_attn_qkvpacked_func', 'flash_attn_kvpacked_func',
         'flash_attn_varlen_qkvpacked_func', 'flash_attn_varlen_kvpacked_func']


def install():
    import flash_attn
    if getattr(flash_attn, "_hexgen2_dispatch_installed", False):
        return
    flash_attn._hexgen2_dispatch_installed = True
    for name in NAMES:
        original = getattr(flash_attn, name)

        def dispatch(*args, _name=name, _original=original, **kwargs):
            q = args[0]
            turing = torch.cuda.get_device_capability(q.device) == (7, 5)
            role = 'decode' if q.ndim == 4 and q.shape[1] == 1 else 'prefill'
            key = ('turing' if turing else 'upstream') + ':' + role
            CALLS[key] = CALLS.get(key, 0) + 1
            if not turing:
                return _original(*args, **kwargs)
            module = importlib.import_module('hexgen2.turing_interface')
            assert q.dtype == torch.float16 and q.shape[-1] == 128
            positional = list(args)
            required = {'flash_attn_qkvpacked_func': 1, 'flash_attn_kvpacked_func': 2,
                        'flash_attn_varlen_qkvpacked_func': 3,
                        'flash_attn_varlen_kvpacked_func': 6}[_name]
            dropout = positional.pop(required) if len(positional) > required else kwargs.pop('dropout_p', 0.0)
            assert dropout == 0.0, 'The Turing backend supports inference with zero dropout only'
            # The Turing extension assumes dense row strides for Q.
            positional = [x.contiguous() if isinstance(x, torch.Tensor) else x for x in positional]
            result = getattr(module, _name)(*positional, **kwargs)
            if AUDIT and _name == 'flash_attn_kvpacked_func':
                query, kv = positional
                k, v = kv.unbind(2)
                expected = torch.nn.functional.scaled_dot_product_attention(query.transpose(1,2).float(), k.transpose(1,2).float(), v.transpose(1,2).float(), is_causal=kwargs.get('causal',False), scale=kwargs.get('softmax_scale')).transpose(1,2)
                ERRORS.append({'max_abs': (result.float()-expected).abs().max().item(), 'qshape': list(query.shape), 'qstride':list(query.stride())})
            return result

        setattr(flash_attn, name, dispatch)


def validate():
    import torch.nn.functional as F
    import flash_attn
    from torch.profiler import profile, ProfilerActivity
    torch.manual_seed(123)
    qkv = torch.randn(1, 32, 3, 20, 128, device='cuda', dtype=torch.float16)
    q, k, v = qkv.unbind(2)
    with torch.inference_mode(), profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        actual = flash_attn.flash_attn_qkvpacked_func(qkv, 0.0, causal=True)
    reference = F.scaled_dot_product_attention(q.transpose(1, 2).float(), k.transpose(1, 2).float(),
        v.transpose(1, 2).float(), is_causal=True).transpose(1, 2).half()
    torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.02)
    kv = torch.stack([k, v], dim=2)
    # Model prefill uses a strided Q view, unlike the packed-QKV path above.
    for length in [23, 256]:
        packed = torch.randn(1, length, 3, 20, 128, device='cuda', dtype=torch.float16) * 3
        qs, ks, vs = packed.unbind(2)
        got = flash_attn.flash_attn_kvpacked_func(qs, torch.stack([ks, vs], 2), 0.0, causal=True)
        expected = F.scaled_dot_product_attention(qs.transpose(1,2).float(), ks.transpose(1,2).float(), vs.transpose(1,2).float(), is_causal=True).transpose(1,2).half()
        torch.testing.assert_close(got, expected, rtol=0.02, atol=0.02)
    decoded = flash_attn.flash_attn_kvpacked_func(q[:, -1:], kv, 0.0, causal=False)
    reference = F.scaled_dot_product_attention(q[:, -1:].transpose(1, 2).float(),
        k.transpose(1, 2).float(), v.transpose(1, 2).float(), is_causal=False).transpose(1, 2).half()
    torch.testing.assert_close(decoded, reference, rtol=0.02, atol=0.02)
    kernels = sorted({e.name for e in prof.events() if str(e.device_type) == 'DeviceType.CUDA'})
    assert kernels, 'No CUDA kernels were observed'
    return {'prefill_and_decode_numerical_checks': 'passed', 'cuda_kernels': kernels,
            'calls': dict(CALLS), 'head_dim': 128, 'dtype': 'float16'}
