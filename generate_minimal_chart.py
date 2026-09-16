import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# Load exact Segoe UI fonts
font_reg = fm.FontProperties(fname='C:/Windows/Fonts/segoeui.ttf')
font_bold = fm.FontProperties(fname='C:/Windows/Fonts/segoeuib.ttf')
font_light = fm.FontProperties(fname='C:/Windows/Fonts/segoeuisl.ttf')

# Data
datasets = ['100,000 items', '500,000 items', '1,000,000 items', '2,000,000 items']
cpu_times = [6.70, 32.50, 63.53, 128.57]       # uFuzzy CPU
gpu_times = [3.33, 3.33, 6.90, 25.20]          # WebGPU RTX 3050
speedups = ['2.0×', '9.8×', '9.2×', '5.1×']

fig, ax = plt.subplots(figsize=(11.5, 6.4), dpi=300, facecolor='#ffffff')
ax.set_facecolor('#ffffff')

y = np.arange(len(datasets))[::-1] # top to bottom
bar_height = 0.28

# Colors: Apple/Linear minimal aesthetic
c_cpu = '#e2e8f0'      # soft, elegant gray
c_gpu = '#2563eb'      # crisp electric blue

# Horizontal bars
bars_cpu = ax.barh(y + bar_height/2 + 0.02, cpu_times, height=bar_height, label='uFuzzy (CPU)', 
                   color=c_cpu, edgecolor='none', zorder=3)
bars_gpu = ax.barh(y - bar_height/2 - 0.02, gpu_times, height=bar_height, label='WebGPU (RTX 3050 Laptop)', 
                   color=c_gpu, edgecolor='none', zorder=3)

# Remove all unnecessary spines
for s in ['top', 'right', 'left', 'bottom']:
    ax.spines[s].set_visible(False)

# Light vertical grid lines only
ax.grid(axis='x', color='#f1f5f9', linestyle='-', linewidth=1.2, zorder=0)
ax.set_axisbelow(True)

# X axis
ax.set_xlim(0, 160)
ax.set_xticks([0, 30, 60, 90, 120, 150])
ax.set_xticklabels(['0 ms', '30 ms', '60 ms', '90 ms', '120 ms', '150 ms'], 
                   fontproperties=font_reg, fontsize=10, color='#94a3b8')
ax.tick_params(axis='both', which='both', length=0, pad=12)

# Y axis
ax.set_yticks(y)
ax.set_yticklabels(datasets, fontproperties=font_bold, fontsize=11, color='#0f172a')

# Labels for bars (absolutely zero overlap possible!)
for bar in bars_cpu:
    w = bar.get_width()
    ax.text(w + 2.5, bar.get_y() + bar.get_height()/2, f'{w:.1f} ms',
            va='center', ha='left', fontproperties=font_reg, fontsize=9.5, color='#64748b')

for i, bar in enumerate(bars_gpu):
    w = bar.get_width()
    ax.text(w + 2.5, bar.get_y() + bar.get_height()/2, f'{w:.1f} ms',
            va='center', ha='left', fontproperties=font_bold, fontsize=9.5, color='#1d4ed8')
    
    # Speedup pill aligned cleanly on the right
    ax.text(142, bar.get_y() + bar.get_height()/2, f'{speedups[i]} faster',
            va='center', ha='center', fontproperties=font_bold, fontsize=9.5, color='#2563eb',
            bbox=dict(boxstyle='round,pad=0.35,rounding_size=0.4', 
                      facecolor='#eff6ff', edgecolor='#bfdbfe', linewidth=0.8))

# Header: Clean typography hierarchy
fig.text(0.08, 0.93, 'WebGPU vs CPU Fuzzy Search', 
         fontproperties=font_bold, fontsize=17, color='#0f172a')
fig.text(0.08, 0.885, 'Query latency in milliseconds (lower is better) • Tested on consumer laptop hardware', 
         fontproperties=font_reg, fontsize=10.5, color='#64748b')

# Minimal legend placed cleanly
leg = ax.legend(loc='lower right', bbox_to_anchor=(0.95, 0.03), frameon=False, 
                prop=font_bold, fontsize=10, labelcolor='#334155')

# Bottom caption
fig.text(0.08, 0.04, 'Query: "AuthController" • CPU: @leeoniya/ufuzzy (single thread) • GPU: NVIDIA RTX 3050 Laptop • Zero server roundtrips',
         fontproperties=font_reg, fontsize=9, color='#94a3b8')

plt.subplots_adjust(top=0.82, bottom=0.13, left=0.18, right=0.95)

out_path = 'C:/Users/ASUS/code/webgpu-fuzzy-search/minimal_benchmark_chart.png'
plt.savefig(out_path, dpi=300, facecolor='#ffffff', edgecolor='none')
plt.close()
print("HORIZONTAL_CHART_OK")
