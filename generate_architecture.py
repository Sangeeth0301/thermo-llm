import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def draw_block(ax, x, y, w, h, text, facecolor='white', edgecolor='black', text_color='black', 
               style="round,pad=0.1,rounding_size=0.15", lw=1.5, ls='-', zorder=2, fontsize=10, italic=False, shadow=True):
    if shadow:
        # Draw soft drop shadow offset
        shadow_box = patches.FancyBboxPatch(
            (x - w/2 + 0.05, y - h/2 - 0.05), w, h,
            boxstyle=style, ec='none', fc='#D0D4DC', alpha=0.6, zorder=zorder-1
        )
        ax.add_patch(shadow_box)
        
    box = patches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=style, ec=edgecolor, fc=facecolor, lw=lw, ls=ls, zorder=zorder
    )
    ax.add_patch(box)
    
    if text:
        weight = 'normal' if italic else 'bold'
        style_t = 'italic' if italic else 'normal'
        ax.text(x, y, text, ha='center', va='center', 
                color=text_color, fontsize=fontsize, fontweight=weight, fontstyle=style_t, family='sans-serif', zorder=zorder+1)
    return (x, y, w, h)

def draw_pill(ax, x, y, w, h, text, facecolor='#FFFFFF', edgecolor='#CCCCCC', text_color='black', fontsize=11):
    return draw_block(ax, x, y, w, h, text, style="round,pad=0.1,rounding_size=0.35", 
                      facecolor=facecolor, edgecolor=edgecolor, text_color=text_color, lw=1.2, fontsize=fontsize, shadow=True)

def draw_arrow(ax, start_x, start_y, end_x, end_y, text=None, text_offset_x=None, text_offset_y=None, color='#2C3E50', lw=1.8):
    ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                arrowprops=dict(arrowstyle='-|>', lw=lw, color=color, shrinkA=0, shrinkB=0, mutation_scale=15),
                zorder=1)
    if text:
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        if text_offset_x is None and text_offset_y is None:
            if abs(start_x - end_x) < 0.1: # Vertical arrow
                text_offset_x = 0.18
                text_offset_y = 0.0
                ha = 'left'
                va = 'center'
            else: # Horizontal arrow
                text_offset_x = 0.0
                text_offset_y = 0.18
                ha = 'center'
                va = 'bottom'
        else:
            text_offset_x = text_offset_x or 0.0
            text_offset_y = text_offset_y or 0.0
            ha = 'center'
            va = 'center'
        ax.text(mid_x + text_offset_x, mid_y + text_offset_y, text, ha=ha, va=va, 
                fontsize=11, fontweight='bold', color=color, family='sans-serif', zorder=4)

def draw_elbow_arrow(ax, start_x, start_y, mid_x, mid_y, end_x, end_y, color='#2C3E50', lw=1.8):
    ax.plot([start_x, mid_x], [start_y, mid_y], color=color, lw=lw, zorder=1)
    draw_arrow(ax, mid_x, mid_y, end_x, end_y, color=color, lw=lw)

def generate_architecture_diagram():
    print("Generating Optimized Colorful System Architecture Diagram...")
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Color Palette
    bg_system1 = '#F0F4F8'    # Soft blue-grey for Observation loop
    border_sys1 = '#D0DBE5'
    bg_system2 = '#F5F3F8'    # Soft purple-grey for Control loop
    border_sys2 = '#E6E1ED'
    bg_offline = '#FFF8F0'    # Soft orange-cream for Calibration
    border_offline = '#FFE8D1'

    # SYSTEM 1 BOUNDS
    draw_block(ax, 8.0, 7.5, 15.0, 4.0, '', facecolor=bg_system1, edgecolor=border_sys1, style="square,pad=0", lw=2, shadow=False)
    ax.text(0.8, 9.15, 'System 1: Cyber-Physical Observation (Telemetry & Thermal Forecasting)', ha='left', va='center', 
            fontsize=14, fontweight='bold', color='#1E293B', family='sans-serif')

    # LLM Workload Block
    draw_block(ax, 1.3, 7.5, 1.4, 1.4, 'LLM\nWorkload\n($w_k$)', facecolor='#334155', edgecolor='#1E293B', text_color='#FFFFFF', lw=1.5, fontsize=12)
    draw_arrow(ax, 2.0, 7.5, 2.6, 7.5)

    # Phase-Aware Runtime Instrumentation
    draw_block(ax, 4.1, 7.5, 3.0, 1.8, 'Phase-Aware Runtime\nInstrumentation\n(llama.cpp Hooks)', facecolor='#FFFFFF', edgecolor='#0288D1', text_color='#01579B', lw=2, fontsize=12)

    # Telemetry Signals
    draw_elbow_arrow(ax, 5.6, 7.5, 5.9, 8.3, 6.1, 8.3)
    draw_pill(ax, 7.2, 8.3, 2.2, 0.7, 'Phase Signal ($\phi$)', facecolor='#E0F7FA', edgecolor='#00ACC1', text_color='#006064')
    
    draw_elbow_arrow(ax, 5.6, 7.5, 5.9, 6.7, 6.1, 6.7)
    draw_pill(ax, 7.2, 6.7, 2.2, 0.7, 'Telemetry ($T, P$)', facecolor='#F1F8E9', edgecolor='#7CB342', text_color='#33691E')

    # Physics-Informed Digital Twin
    draw_block(ax, 10.9, 7.5, 3.6, 2.2, 'Physics-Informed Digital Twin\n(PINN Neural-ROM)\n[10s Forecast Horizon]', facecolor='#FFFFFF', edgecolor='#E64A19', text_color='#BF360C', lw=2, fontsize=12)
    
    # Connecting Phase & Telemetry to Twin
    draw_elbow_arrow(ax, 8.3, 8.3, 8.6, 7.7, 9.1, 7.7)
    draw_arrow(ax, 8.3, 6.7, 9.1, 6.7)

    # Offline Calibration Box
    draw_block(ax, 12.0, 10.6, 6.8, 1.6, '', facecolor=bg_offline, edgecolor=border_offline, style="square,pad=0", lw=1.5, ls='--', shadow=False)
    ax.text(8.8, 11.1, 'Offline Calibration Phase', ha='left', va='center', fontsize=12, fontweight='bold', color='#D84315')
    
    draw_pill(ax, 9.8, 10.4, 2.0, 0.6, 'Dataset ($\mathcal{D}_{\mathrm{thermo}}$)', facecolor='#FFFFFF', edgecolor='#FF9800', text_color='#E65100', fontsize=10)
    draw_pill(ax, 12.0, 10.4, 2.0, 0.6, 'Parameter Estimation\n($R_{\mathrm{th}}, C_{\mathrm{th}}$)', facecolor='#FFFFFF', edgecolor='#FF9800', text_color='#E65100', fontsize=9)
    draw_pill(ax, 14.2, 10.4, 2.0, 0.6, 'PINN Offline Training', facecolor='#FFFFFF', edgecolor='#FF9800', text_color='#E65100', fontsize=10)
    
    draw_arrow(ax, 10.8, 10.4, 11.0, 10.4)
    draw_arrow(ax, 13.0, 10.4, 13.2, 10.4)
    
    # Path from Offline Training to Online Twin
    draw_elbow_arrow(ax, 14.2, 10.1, 14.2, 8.3, 12.7, 8.3, color='#E64A19')
    ax.text(13.6, 8.4, 'Frozen Weights', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#E64A19')

    # SYSTEM 2 BOUNDS
    draw_block(ax, 8.0, 2.4, 15.0, 4.6, '', facecolor=bg_system2, edgecolor=border_sys2, style="square,pad=0", lw=2, shadow=False)
    ax.text(0.8, 4.4, 'System 2: Safety-Constrained Closed-Loop Control', ha='left', va='center', fontsize=14, fontweight='bold', color='#1E293B', family='sans-serif')

    # State Vector
    draw_pill(ax, 7.2, 2.8, 3.2, 1.2, 'State Vector\n$\mathbf{s}_t = (T, P, f, \phi, \mathbf{\hat{T}})$', facecolor='#E0F2F1', edgecolor='#00897B', text_color='#004D40', fontsize=12)
    
    # Phase down to State Vector
    draw_arrow(ax, 7.2, 7.95, 7.2, 3.4, color='#004D40')
    
    # Twin down to State Vector (Forecast)
    draw_elbow_arrow(ax, 10.9, 6.4, 10.9, 4.6, 7.2, 4.6, color='#00897B')
    draw_arrow(ax, 7.2, 4.6, 7.2, 3.4, color='#00897B')
    ax.text(9.05, 4.7, 'Multi-step Thermal Forecast ($\mathbf{\hat{T}}$)', ha='center', va='bottom', fontsize=11, fontweight='bold', color='#00897B')

    # D-DQN Scheduler
    draw_block(ax, 10.7, 2.8, 2.4, 1.6, 'D-DQN\nScheduler\n(RL Agent)', facecolor='#FFFFFF', edgecolor='#7B1FA2', text_color='#4A148C', lw=2, fontsize=12)
    draw_arrow(ax, 8.8, 2.8, 9.5, 2.8) # state -> DQN

    # Safety Shield
    draw_block(ax, 13.9, 2.8, 2.6, 1.6, 'Deterministic\nSafety Shield\n(Veto & Action Repair)', facecolor='#FFFFFF', edgecolor='#388E3C', text_color='#1B5E20', lw=2, fontsize=12)
    draw_arrow(ax, 11.9, 2.8, 12.6, 2.8, text='$f_{\mathrm{cand}}$')

    # Twin straight down to Shield (Verification Forecast)
    draw_elbow_arrow(ax, 12.4, 6.4, 12.4, 4.6, 13.9, 4.6, color='#BF360C')
    draw_arrow(ax, 13.9, 4.6, 13.9, 3.6, color='#BF360C')
    ax.text(13.15, 4.7, 'Candidate Forecast', ha='center', va='bottom', fontsize=11, fontweight='bold', color='#BF360C')

    # Hardware Resource Controller
    draw_block(ax, 13.9, 0.5, 3.4, 0.8, 'Hardware Resource Controller\n(DVFS Kernel Driver)', facecolor='#334155', edgecolor='#1E293B', text_color='#FFFFFF', lw=1.5, fontsize=11)
    draw_arrow(ax, 13.9, 2.0, 13.9, 0.9, text='Safe Action ($f^*$)', color='#1B5E20')

    path = os.path.join(OUTPUT_DIR, 'fig2_architecture.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    parent_path = os.path.join(os.path.dirname(OUTPUT_DIR), 'fig2_architecture.png')
    plt.savefig(parent_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"-> Saved: {path} and {parent_path}")

if __name__ == '__main__':
    generate_architecture_diagram()
