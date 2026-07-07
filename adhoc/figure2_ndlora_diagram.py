#!/usr/bin/env python3
"""
Generate ND-LoRA architecture diagram for Figure 3
"""

import logging
from itertools import combinations
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, FancyBboxPatch

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

# Component tracking class

TEXT_KWARGS = dict(
    ha='center',
    va='center',
    fontsize=9,
    fontweight='bold',
    linespacing=1.5,
)


class Component:
    def __init__(self, x, y, width, height, padding=0.0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.padding = padding

    @property
    def left_center(self):
        return (self.x - self.padding, self.y + self.height / 2)

    @property
    def right_center(self):
        return (self.x + self.width + self.padding, self.y + self.height / 2)

    @property
    def top_center(self):
        return (self.x + self.width / 2, self.y + self.height + self.padding)

    @property
    def bottom_center(self):
        return (self.x + self.width / 2, self.y - self.padding)

    @property
    def center(self):
        return (self.x + self.width / 2, self.y + self.height / 2)


# Set up the figure (wider for left-to-right flow)
fig, ax = plt.subplots(1, 1, figsize=(11.2, 5))  # Reduced height to 5.5 (removed title space)
ax.set_xlim(0, 11.2)
ax.set_ylim(0, 5.5)
ax.axis('off')

# Colors
stream_colors = ['#E8F4FD', '#FFE0B3', '#D5E8D4', '#E6D5F5']  # Light blue, light apricot, green, light purple
stream_text_colors = ['#2563B8', '#CC8800', '#388E3C', '#7B4BA6']  # Bright, saturated versions of stream colors (for text)
lora_color = '#FFE6CC'  # Light orange
aggregator_color = '#FFF2CC'  # Light yellow (swapped with stream 2)
bt_color = '#FFE6E6'  # Light red/pink for Barlow Twins
bt_border_color = '#CC0000'  # Dark red border

# Component registry
components = {}

# Stream processing boxes (vertical stacking, left-to-right flow with better spacing)
stream_y_positions = [4.5, 3.2, 1.9, 0.6]
stream_width = 3.2  # Reduced by 20% from 4.0 to 3.2

# Calculate vertical center based on stream positions
stream_y_min = min(stream_y_positions)
stream_y_max = max(stream_y_positions)
y_center = (stream_y_min + stream_y_max) / 2

# Input (left side, vertically centered relative to streams)
input_x, input_w, input_h = 0.2, 0.9, 1
input_y = y_center - input_h / 2
components['input'] = Component(input_x, input_y, input_w, input_h, padding=0.1)
input_box = FancyBboxPatch((input_x, input_y), input_w, input_h,
                           boxstyle="round,pad=0.1",
                           facecolor='white',
                           edgecolor='black',
                           linewidth=2)
ax.add_patch(input_box)
ax.text(*components['input'].center, 'Input\nTokens', **TEXT_KWARGS)

components['prefixes'] = []
components['streams'] = []
components['loras'] = []

for i, y in enumerate(stream_y_positions):
    # Prefix tokens (left of each stream)
    prefix_x, prefix_y, prefix_w, prefix_h = input_x + input_w + 0.5, y - 0.25, 1, 0.5
    components['prefixes'].append(Component(prefix_x, prefix_y, prefix_w, prefix_h, padding=0.05))
    prefix_box = FancyBboxPatch((prefix_x, prefix_y), prefix_w, prefix_h,
                                boxstyle="round,pad=0.05",
                                facecolor=stream_colors[i],
                                edgecolor='black',
                                linewidth=2)
    ax.add_patch(prefix_box)
    ax.text(prefix_x + prefix_w/2, y, f'Stream #{i+1}\nPrefix', **TEXT_KWARGS)

    # Stream box (horizontal, contains Qwen2.5 → LoRA)
    stream_x, stream_y, stream_w, stream_h = prefix_x + prefix_w + 0.5, y - 0.35, stream_width, 0.7
    components['streams'].append(Component(stream_x, stream_y, stream_w, stream_h, padding=0.1))
    stream_box = FancyBboxPatch((stream_x, stream_y), stream_w, stream_h,
                                boxstyle="round,pad=0.1",
                                facecolor='#D3D3D3',
                                edgecolor='black',
                                linewidth=2,
                                zorder=2)
    ax.add_patch(stream_box)
    # ax.text(stream_x + 0.67, y, f'Stream #{i+1}', ha='center', va='center', color=stream_text_colors[i],
    #         fontsize=10, fontweight='bold', zorder=15)

    # LoRA adapter (right side of stream)
    lora_x, lora_y, lora_w, lora_h = stream_x + stream_w - 1.1, y - 0.25, 1, 0.5
    components['loras'].append(Component(lora_x, lora_y, lora_w, lora_h, padding=0.05))
    lora_box = FancyBboxPatch((lora_x, lora_y), lora_w, lora_h,
                              boxstyle="round,pad=0.05",
                              facecolor=stream_colors[i],
                              edgecolor="black",
                              linewidth=1.5,
                              zorder=3)
    ax.add_patch(lora_box)
    ax.text(lora_x + lora_w/2, y, f'Stream #{i+1}\nLoRA', **TEXT_KWARGS)

# Calculate Qwen2.5 module position (drawn first, before streams)
# Convert pixels to data coordinates (approximately, 72 DPI, figsize 11.2x6)
px_to_data_x = 11.2 / (11.2 * 72)  # data units per pixel
px_to_data_y = 6.0 / (6 * 72)  # data units per pixel

# Position: ~100px right of "Stream N" text (which is at stream_x_start + 0.5)
qwen_x = stream_x + 10 * px_to_data_x
# Width: space between Stream N and LoRA boxes, minus 100px on each side
qwen_w = lora_x - qwen_x - 50 * px_to_data_x
# Height: from 100px above bottom of last stream to 100px below top of first stream
qwen_y_top = stream_y_positions[0] + 35 * px_to_data_y
qwen_y_bottom = stream_y_positions[-1] - 35 * px_to_data_y
qwen_y = qwen_y_bottom
qwen_h = qwen_y_top - qwen_y_bottom

qwen_zorder = 10
qwen_alpha = 0.85
components['qwen_model'] = Component(qwen_x, qwen_y, qwen_w, qwen_h, padding=0.0)
qwen_box = FancyBboxPatch((qwen_x, qwen_y), qwen_w, qwen_h,
                          boxstyle="round,pad=0.1",
                          facecolor='#D3D3D3',  # Gray
                          edgecolor='#808080',  # Darker gray border
                          linewidth=2.5,
                          alpha=qwen_alpha,
                          zorder=qwen_zorder)
ax.add_patch(qwen_box)
ax.text(qwen_x + qwen_w/2, y_center, 'Frozen\nShared\nBackbone', zorder=qwen_zorder, color='#404040',
        **TEXT_KWARGS)

# Aggregator (receives stream outputs, vertically centered)
agg_x, agg_w, agg_h = stream_x + stream_w + 0.75, 1.25, 2.5  # Moved left from 10.5 to 8.5
agg_y = y_center - agg_h / 2
components['aggregator'] = Component(agg_x, agg_y, agg_w, agg_h, padding=0.1)
agg_box = FancyBboxPatch((agg_x, agg_y), agg_w, agg_h, zorder=5,
                         boxstyle="round,pad=0.1",
                         facecolor=aggregator_color,
                         edgecolor='black',
                         linewidth=2)
ax.add_patch(agg_box)
ax.text(agg_x + agg_w/2, y_center, 'Multi-Stream\nAggregator', zorder=10, **TEXT_KWARGS)
# ax.text(agg_x + agg_w/2, y_center - 0.1, r'$\alpha_i \sim Softmax(MLP)$',
#         ha='center', va='center', zorder=10, fontsize=10, fontweight='bold')

# LM Head (AFTER aggregator, vertically centered)
lm_head_x, lm_head_w, lm_head_h = agg_x + agg_w + 0.5, 0.75, 0.8  # Reduced width by 20% from 1.3 to 1.04
lm_head_y = y_center - lm_head_h / 2
components['lm_head'] = Component(lm_head_x, lm_head_y, lm_head_w, lm_head_h, padding=0.1)
lm_head_box = FancyBboxPatch((lm_head_x, lm_head_y), lm_head_w, lm_head_h,
                             boxstyle="round,pad=0.1",
                             facecolor='lightgray',
                             edgecolor='black',
                             linewidth=2)
ax.add_patch(lm_head_box)
ax.text(*components['lm_head'].center, 'LM\nHead', **TEXT_KWARGS)

# Output (rightmost, vertically centered)
output_x, output_w, output_h = lm_head_x + lm_head_w + 0.5, 0.9, 0.8  # Reduced width by 20% from 1.2 to 0.96
output_y = y_center - output_h / 2
components['output'] = Component(output_x, output_y, output_w, output_h, padding=0.1)
output_box = FancyBboxPatch((output_x, output_y), output_w, output_h,
                            boxstyle="round,pad=0.1",
                            facecolor='white',
                            edgecolor='black',
                            linewidth=2)
ax.add_patch(output_box)
ax.text(*components['output'].center, 'Output\nTokens', **TEXT_KWARGS)

# Barlow Twins regularization box (top right)
bt_x, bt_w, bt_y, bt_h = agg_x, 3, 4.3, 1.0
components['bt'] = Component(bt_x, bt_y, bt_w, bt_h, padding=0.1)
bt_box = FancyBboxPatch((bt_x, bt_y), bt_w, bt_h,
                        boxstyle="round,pad=0.1",
                        facecolor=bt_color,  # Light red/pink to match red dashed lines
                        edgecolor=bt_border_color,  # Darker red border
                        linewidth=2.5,
                        linestyle='--')
ax.add_patch(bt_box)
ax.text(bt_x + bt_w/2, bt_y + bt_h - 0.2, 'Barlow Twins\nRegularization', **TEXT_KWARGS)
ax.text(bt_x + bt_w/2, bt_y + 0.2, r'$\mathcal{L}_{BT} = \frac{1}{P(P-1)} \sum_{i \neq j} \|C_{ij} - I\|_F^2$',
        **TEXT_KWARGS)


# Arrows - Left to right flow (starting/ending at borders)
# Input to streams (solid arrows, from input right border to stream left border)
for i, stream in enumerate(components['streams']):
    arrow = ConnectionPatch(components['input'].right_center, stream.left_center, "data", "data",
                            arrowstyle="->", shrinkA=0, shrinkB=0,
                            lw=2, color='black', alpha=0.7, zorder=0)
    ax.add_artist(arrow)

# Prefix to streams (dotted arrows, from prefix right border to stream left border)
for i, (prefix, stream) in enumerate(zip(components['prefixes'], components['streams'])):
    arrow = ConnectionPatch(prefix.right_center, stream.left_center, "data", "data",
                            arrowstyle="->", shrinkA=0, shrinkB=0,
                            lw=2, color='black', alpha=0.7, zorder=0)
    ax.add_artist(arrow)

# Streams to aggregator (solid arrows from stream right border to aggregator left border)
for i, stream in enumerate(components['streams']):
    arrow = ConnectionPatch(stream.right_center, components['aggregator'].left_center, "data", "data",
                            arrowstyle="->", shrinkA=0, shrinkB=0,
                            lw=2, color='black', alpha=0.7, zorder=0)
    ax.add_artist(arrow)

# Pairwise de-correlation arrows between LoRA boxes (curved, red dashed)
for (i, lora1), (j, lora2) in combinations(enumerate(components['loras']), 2):
    arrow = ConnectionPatch(
        lora1.left_center, lora2.left_center, "data", "data",
        arrowstyle="<|-|>", mutation_scale=12, linestyle=(0, (1, 2)),
        lw=2, color='red', alpha=0.7, zorder=15,
        connectionstyle="arc3,rad=0.250",
    )
    ax.add_artist(arrow)

# LoRA boxes to BT box (dotted red lines, from LoRA center to BT bottom border)
for i, lora in enumerate(components['loras']):
    line = ConnectionPatch(lora.right_center, components['bt'].left_center, "data", "data",
                           arrowstyle="-", shrinkA=0, shrinkB=0, zorder=4,
                           linestyle=':', lw=2, color='red', alpha=0.7)
    ax.add_artist(line)

# Aggregator to LM Head (from agg right border to lm_head left border)
arrow = ConnectionPatch(components['aggregator'].right_center, components['lm_head'].left_center, "data", "data",
                        arrowstyle="->", shrinkA=0, shrinkB=0,
                        lw=2, color='black', alpha=0.7, zorder=0)
ax.add_artist(arrow)

# LM Head to Output (from lm_head right border to output left border)
arrow = ConnectionPatch(components['lm_head'].right_center, components['output'].left_center, "data", "data",
                        arrowstyle="->", shrinkA=0, shrinkB=0,
                        lw=2, color='black', alpha=0.7, zorder=0)
ax.add_artist(arrow)

# Add title
TEXT_KWARGS["fontsize"] = 18
title_x = (output_x + output_w) / 2
ax.text(title_x, 5.9, 'ND-LoRA Architecture (P=4)', **TEXT_KWARGS)
plt.tight_layout()

basedir = Path(__file__).parent.parent / "paper" / "assets"
plt.savefig(basedir / 'figure2_diagram.pdf', bbox_inches='tight', dpi=600, format='pdf')
plt.savefig(basedir / 'figure2_diagram.png', bbox_inches='tight', dpi=600, format='png')

logging.info("ND-LoRA architecture diagram saved %s/figure2_diagram.*", basedir)
