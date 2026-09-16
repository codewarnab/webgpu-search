import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# Load exact Segoe UI fonts
font_reg = fm.FontProperties(fname='C:/Windows/Fonts/segoeui.ttf')
font_bold = fm.FontProperties(fname='C:/Windows/Fonts/segoeuib.ttf')

# Data
datasets = ['100,000\nitems', '500,000\nitems', '1,000,000\nitems', '2,000,000\nitems']
cpu_times = [6.70, 32.50, 63.53, 128.57]       # uFuzzy CPU
gpu_times = [3.33, 3.33, 6.90, 25.20]          # WebGPU RTX 3050
speedups = ['2.0×', '9.8×', '9.2×', '5.1×']

fig, ax = plt.subplots(figsize=(11, 6.4), dpi=300, facecolor='#ffffff')
ax.set_facecolor('#ffffff')

x = np.arange(len(datasets))
width = 0.30

c_cpu = '#e2e8f0'      # soft minimal slate
c_gpu = '#2563eb'      # clean electric royal blue

# Vertical bars
bars_cpu = ax.bar(x - width/2 - 0.02, cpu_times, width=width, label='uFuzzy (Single-thread CPU)', 
                  color=c_cpu, edgecolor='none', zorder=3)
bars_gpu = ax.bar(x + width/2 + 0.02, gpu_times, width=width, label='WebGPU (Hardware Accelerated)', 
                  color=c_gpu, edgecolor='none', zorder=3)

# Remove unnecessary spines
for s in ['top', 'right', 'left']:
    ax.spines[s].set_visible(False)
ax.spines['bottom'].set_color('#e2e8f0')
ax.spines['bottom'].set_linewidth(1.2)

# Subtle horizontal grid only
ax.grid(axis='y', color='#f8fafc', linestyle='-', linewidth=1.2, zorder=0)
ax.set_axisbelow(True)

# Y axis limits and ticks
ax.set_ylim(0, 160)
ax.set_yticks([0, 30, 60, 90, 120, 150])
ax.set_yticklabels(['0 ms', '30 ms', '60 ms', '90 ms', '120 ms', '150 ms'], 
                   fontproperties=font_reg, fontsize=10, color='#94a3b8')
ax.tick_params(axis='both', which='both', length=0, pad=10)

# X axis
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontproperties=font_bold, fontsize=11, color='#0f172a')

# Numbers above CPU bars
for bar in bars_cpu:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 3.5, f'{h:.1f}ms',
            ha='center', va='bottom', fontproperties=font_reg, fontsize=10, color='#64748b')

# Numbers & speedup pills above WebGPU bars (NO overlapping lines, ample spacing)
for i, bar in enumerate(bars_gpu):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 3.5, f'{h:.1f}ms',
            ha='center', va='bottom', fontproperties=font_bold, fontsize=10, color='#1d4ed8')
    
    # Speedup badge cleanly floating
    ax.text(bar.get_x() + bar.get_width()/2, h + 15.0, f'{speedups[i]} faster',
            ha='center', va='bottom', fontproperties=font_bold, fontsize=9.5, color='#2563eb',
            bbox=dict(boxstyle='round,pad=0.32,rounding_size=0.4', 
                      facecolor='#eff6ff', edgecolor='#bfdbfe', linewidth=0.8))

# Header: Clean typography hierarchy
fig.text(0.08, 0.93, 'WebGPU vs CPU Fuzzy Search', 
         fontproperties=font_bold, fontsize=17, color='#0f172a')
fig.text(0.08, 0.885, 'Query latency in milliseconds (lower is better) • Tested on consumer laptop hardware', 
         fontproperties=font_reg, fontsize=10.5, color='#64748b')

# Minimal legend placed in top-left open area
ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.96), frameon=False, 
          prop=font_bold, fontsize=10, labelcolor='#334155')

# Bottom caption / context
fig.text(0.08, 0.04, 'Query: "AuthController" • CPU: @leeoniya/ufuzzy (single thread) • GPU: NVIDIA RTX 3050 Laptop • Zero server roundtrips',
         fontproperties=font_reg, fontsize=9, color='#94a3b8')

plt.subplots_adjust(top=0.82, bottom=0.14, left=0.08, right=0.94)

out_path = 'C:/Users/ASUS/code/webgpu-fuzzy-search/minimal_vertical_chart.png'
plt.savefig(out_path, dpi=300, facecolor='#ffffff', edgecolor='none')
plt.close()
print("VERTICAL_CHART_OK")
