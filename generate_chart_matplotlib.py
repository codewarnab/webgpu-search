import matplotlib.pyplot as plt
import numpy as np

# Set style & typography
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.linewidth'] = 1.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.2), dpi=300, facecolor='#090a0f')

# Benchmark data from CSVs
datasets = ['100K', '500K', '1.0M', '2.0M']
cpu_times = [6.70, 32.50, 63.53, 128.57]           # uFuzzy (CPU)
gpu_rtx_times = [3.33, 3.33, 6.90, 25.20]         # RTX 3050 Laptop
gpu_intel_times = [3.00, 5.27, 27.70, 15.10]      # Intel Iris Xe (Integrated)

speedup_rtx = [cpu_times[i] / gpu_rtx_times[i] for i in range(len(datasets))]
speedup_intel = [cpu_times[i] / gpu_intel_times[i] for i in range(len(datasets))]

x = np.arange(len(datasets))
width = 0.26

# ----------------- PANEL 1: Search Latency (ms) -----------------
ax1.set_facecolor('#0f172a')
rects1 = ax1.bar(x - width, cpu_times, width, label='uFuzzy CPU', color='#475569', edgecolor='#64748b', linewidth=0.8)
rects2 = ax1.bar(x, gpu_intel_times, width, label='WebGPU (Intel Iris Xe)', color='#0284c7', edgecolor='#38bdf8', linewidth=0.8)
rects3 = ax1.bar(x + width, gpu_rtx_times, width, label='WebGPU (RTX 3050 Laptop)', color='#10b981', edgecolor='#34d399', linewidth=0.8)

# 16ms frame budget threshold line
ax1.axhline(16.6, color='#ef4444', linestyle='--', linewidth=1.2, alpha=0.8, label='60 FPS Frame Budget (16.6ms)')

ax1.set_title('Query Latency by Dataset Size\n(Lower is Better • Query: "AuthController")', color='#f8fafc', fontsize=13, fontweight='bold', pad=14)
ax1.set_xlabel('Number of Indexed Strings', color='#94a3b8', fontsize=11, fontweight='semibold', labelpad=8)
ax1.set_ylabel('Execution Time (ms)', color='#94a3b8', fontsize=11, fontweight='semibold', labelpad=8)
ax1.set_xticks(x)
ax1.set_xticklabels(datasets, color='#e2e8f0', fontsize=10, fontweight='bold')
ax1.tick_params(colors='#94a3b8', which='both')
ax1.grid(axis='y', linestyle=':', color='#1e293b', alpha=0.9)

# Value annotations on latency bars
for bar in rects1:
    h = bar.get_height()
    ax1.annotate(f'{h:.1f}ms', xy=(bar.get_x() + bar.get_width() / 2, h),
                 xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                 fontsize=8.5, color='#94a3b8', fontweight='bold')

for bar in rects2:
    h = bar.get_height()
    ax1.annotate(f'{h:.1f}ms', xy=(bar.get_x() + bar.get_width() / 2, h),
                 xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                 fontsize=8.5, color='#38bdf8', fontweight='bold')

for bar in rects3:
    h = bar.get_height()
    ax1.annotate(f'{h:.1f}ms', xy=(bar.get_x() + bar.get_width() / 2, h),
                 xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                 fontsize=8.5, color='#34d399', fontweight='bold')

leg1 = ax1.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=9, loc='upper left')

# ----------------- PANEL 2: Speedup Multiple (X-times Faster) -----------------
ax2.set_facecolor('#0f172a')
rects_s1 = ax2.bar(x - width/2, speedup_intel, width, label='WebGPU (Intel Iris Xe)', color='#0284c7', edgecolor='#38bdf8', linewidth=0.8)
rects_s2 = ax2.bar(x + width/2, speedup_rtx, width, label='WebGPU (RTX 3050 Laptop)', color='#10b981', edgecolor='#34d399', linewidth=0.8)

# 1.0x baseline line
ax2.axhline(1.0, color='#94a3b8', linestyle=':', linewidth=1.2, alpha=0.8, label='CPU Baseline (1.0x)')

ax2.set_title('Speedup Multiplier vs CPU uFuzzy\n(Higher is Better • Measured Across Consumer Laptops)', color='#f8fafc', fontsize=13, fontweight='bold', pad=14)
ax2.set_xlabel('Number of Indexed Strings', color='#94a3b8', fontsize=11, fontweight='semibold', labelpad=8)
ax2.set_ylabel('Speedup Factor (x)', color='#94a3b8', fontsize=11, fontweight='semibold', labelpad=8)
ax2.set_xticks(x)
ax2.set_xticklabels(datasets, color='#e2e8f0', fontsize=10, fontweight='bold')
ax2.tick_params(colors='#94a3b8', which='both')
ax2.set_ylim(0, 11.5)
ax2.grid(axis='y', linestyle=':', color='#1e293b', alpha=0.9)

# Value annotations on speedup bars
for bar in rects_s1:
    h = bar.get_height()
    ax2.annotate(f'{h:.1f}x', xy=(bar.get_x() + bar.get_width() / 2, h),
                 xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                 fontsize=9, color='#38bdf8', fontweight='bold')

for bar in rects_s2:
    h = bar.get_height()
    ax2.annotate(f'{h:.1f}x', xy=(bar.get_x() + bar.get_width() / 2, h),
                 xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                 fontsize=9, color='#34d399', fontweight='bold')

leg2 = ax2.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=9, loc='upper left')

# Main title
fig.suptitle('In-Browser Fuzzy Search Benchmark: WebGPU vs CPU', color='#ffffff', fontsize=18, fontweight='bold', y=0.98)

# Subtitle / Footer metadata
fig.text(0.5, 0.02, 'Tested on Google Chrome (Windows 11) • Retained GPU Buffers • Zero Backend Latency', 
         ha='center', color='#64748b', fontsize=10)

plt.tight_layout(rect=[0, 0.04, 1, 0.94])

output_path = 'C:/Users/ASUS/code/webgpu-fuzzy-search/matplotlib_benchmark_chart.png'
plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=300)
plt.close()
print("MATPLOTLIB_CHART_SAVED_AT:", output_path)
