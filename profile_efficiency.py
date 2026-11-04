import torch
import numpy as np
import time
from thop import profile

from model_baseline import build_model as build_baseline
from model import build_model as build_cbam
from model_ecanet import build_model as build_eca

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Profiling on device:', device)
if device.type == 'cuda':
    print('GPU Name:', torch.cuda.get_device_name(0))
    print(f'VRAM Total: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB')

models = {
    'Baseline': build_baseline(),
    'Hybrid (CBAM)': build_cbam(),
    'Hybrid (ECA-Net)': build_eca(),
}

results = {}

for name, m in models.items():
    m.eval()
    m.to(device)
    dummy_1 = torch.randn(1, 3, 224, 224, device=device)
    dummy_32 = torch.randn(32, 3, 224, 224, device=device)
    
    # Total and Trainable parameters
    total_params = sum(p.numel() for p in m.parameters())
    trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    
    # Attention module parameter breakdown
    att_params = 0
    if name in ['Hybrid (CBAM)', 'Hybrid (ECA-Net)']:
        for idx in [5, 6, 7]:
            att_layer = m.features[idx][1]
            att_params += sum(p.numel() for p in att_layer.parameters())
            
    # Model size in MB
    param_size = sum(p.numel() * p.element_size() for p in m.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in m.buffers())
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    # FLOPs & MACs (for single image 1x3x224x224)
    m_cpu = models[name].cpu()
    d_cpu = torch.randn(1, 3, 224, 224)
    macs, _ = profile(m_cpu, inputs=(d_cpu,), verbose=False)
    flops = macs * 2
    m.to(device)
    
    # Latency & Throughput (Batch Size = 1)
    for _ in range(50):
        with torch.no_grad():
            _ = m(dummy_1)
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    reps = 300
    timings = []
    if device.type == 'cuda':
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            for _ in range(reps):
                starter.record()
                _ = m(dummy_1)
                ender.record()
                torch.cuda.synchronize()
                timings.append(starter.elapsed_time(ender))
    else:
        with torch.no_grad():
            for _ in range(reps):
                t0 = time.perf_counter()
                _ = m(dummy_1)
                t1 = time.perf_counter()
                timings.append((t1 - t0) * 1000)
                
    lat_mean = np.mean(timings)
    lat_std = np.std(timings)
    fps_bs1 = 1000.0 / lat_mean
    
    # Latency & Throughput (Batch Size = 32)
    for _ in range(20):
        with torch.no_grad():
            _ = m(dummy_32)
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    timings_32 = []
    if device.type == 'cuda':
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            for _ in range(100):
                starter.record()
                _ = m(dummy_32)
                ender.record()
                torch.cuda.synchronize()
                timings_32.append(starter.elapsed_time(ender))
    else:
        with torch.no_grad():
            for _ in range(100):
                t0 = time.perf_counter()
                _ = m(dummy_32)
                t1 = time.perf_counter()
                timings_32.append((t1 - t0) * 1000)
                
    lat32_mean = np.mean(timings_32)
    fps_bs32 = (32 * 1000.0) / lat32_mean
    
    results[name] = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'att_params': att_params,
        'size_mb': size_mb,
        'macs': macs,
        'flops': flops,
        'lat_mean': lat_mean,
        'lat_std': lat_std,
        'fps_bs1': fps_bs1,
        'fps_bs32': fps_bs32
    }

print('\n' + '='*85)
print(f"{'Efficiency Metric':<35} | {'Baseline':<14} | {'Hybrid (CBAM)':<15} | {'Hybrid (ECA-Net)':<15}")
print('='*85)

b = results['Baseline']
c = results['Hybrid (CBAM)']
e = results['Hybrid (ECA-Net)']

rows = [
    ("Total Parameters", f"{b['total_params']:,}", f"{c['total_params']:,}", f"{e['total_params']:,}"),
    ("Attention Module Params", f"{b['att_params']:,}", f"{c['att_params']:,}", f"{e['att_params']:,}"),
    ("Added Params vs Baseline", "0 (+0.00%)", f"+{c['total_params'] - b['total_params']:,} (+{((c['total_params']-b['total_params'])/b['total_params'])*100:.2f}%)", f"+{e['total_params'] - b['total_params']:,} (+{((e['total_params']-b['total_params'])/b['total_params'])*100:.4f}%)"),
    ("Model Size (Memory/Disk)", f"{b['size_mb']:.2f} MB", f"{c['size_mb']:.2f} MB", f"{e['size_mb']:.2f} MB"),
    ("MACs (per image)", f"{b['macs']/1e6:.2f} M", f"{c['macs']/1e6:.2f} M", f"{e['macs']/1e6:.2f} M"),
    ("FLOPs (per image)", f"{b['flops']/1e6:.2f} M", f"{c['flops']/1e6:.2f} M", f"{e['flops']/1e6:.2f} M"),
    ("Added FLOPs vs Baseline", "0 (+0.00%)", f"+{(c['flops'] - b['flops'])/1e6:.2f} M (+{((c['flops']-b['flops'])/b['flops'])*100:.2f}%)", f"+{(e['flops'] - b['flops'])/1e6:.4f} M (+{((e['flops']-b['flops'])/b['flops'])*100:.4f}%)"),
    ("Inference Latency (BS=1)", f"{b['lat_mean']:.2f} ± {b['lat_std']:.2f} ms", f"{c['lat_mean']:.2f} ± {c['lat_std']:.2f} ms", f"{e['lat_mean']:.2f} ± {e['lat_std']:.2f} ms"),
    ("Throughput (BS=1)", f"{b['fps_bs1']:.1f} FPS", f"{c['fps_bs1']:.1f} FPS", f"{e['fps_bs1']:.1f} FPS"),
    ("Throughput (BS=32)", f"{b['fps_bs32']:.1f} FPS", f"{c['fps_bs32']:.1f} FPS", f"{e['fps_bs32']:.1f} FPS"),
]

for label, bv, cv, ev in rows:
    print(f"{label:<35} | {bv:<14} | {cv:<15} | {ev:<15}")
print('='*85)
