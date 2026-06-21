import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def draw_block(ax, x, y, w, h, text, facecolor='white', edgecolor='black', text_color='black', 
               style="round,pad=0.1,rounding_size=0.1", lw=1.5, ls='-', zorder=2, fontsize=10, italic=False):
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

def draw_pill(ax, x, y, w, h, text, fontsize=12):
    return draw_block(ax, x, y, w, h, text, style="round,pad=0.1,rounding_size=0.4", facecolor='#FFFFFF', lw=1.2, fontsize=fontsize)

def draw_arrow(ax, start_x, start_y, end_x, end_y, text=None, text_offset_x=None, text_offset_y=None):
    ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                arrowprops=dict(arrowstyle='-|>', lw=1.5, color='black', shrinkA=0, shrinkB=0),
                zorder=1)
    if text:
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        if text_offset_x is None and text_offset_y is None:
            if abs(start_x - end_x) < 0.1: # Vertical arrow
                text_offset_x = 0.15
                text_offset_y = 0.0
                ha = 'left'
                va = 'center'
            else: # Horizontal arrow
                text_offset_x = 0.0
                text_offset_y = 0.15
                ha = 'center'
                va = 'bottom'
        else:
            text_offset_x = text_offset_x or 0.0
            text_offset_y = text_offset_y or 0.0
            ha = 'center'
            va = 'center'
        ax.text(mid_x + text_offset_x, mid_y + text_offset_y, text, ha=ha, va=va, fontsize=12, fontweight='bold', family='sans-serif', zorder=4)

def draw_elbow_arrow(ax, start_x, start_y, mid_x, mid_y, end_x, end_y):
    ax.plot([start_x, mid_x], [start_y, mid_y], color='black', lw=1.5, zorder=1)
    draw_arrow(ax, mid_x, mid_y, end_x, end_y)

def generate_architecture_diagram():
    print("Generating Spaced B&W Architecture Diagram...")
    fig, ax = plt.subplots(figsize=(16, 12.0))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 12.0)
    ax.axis('off')

    bg_system = '#F0F0F0'
    bg_offline = '#EAEAEA'

    # SYSTEM 1
    draw_block(ax, 7.5, 7.5, 14.5, 4.0, '', facecolor=bg_system, edgecolor='#888888', style="square,pad=0", lw=1.5, zorder=0)
    ax.text(0.5, 9.2, 'System 1: Cyber-Physical Observation (Runtime & Physics)', ha='left', va='center', fontsize=14, fontweight='bold')

    # LLM
    draw_block(ax, 1.0, 7.5, 1.2, 1.5, 'LLM\nWorkload\n($w_k$)', lw=1.5, fontsize=12)
    draw_arrow(ax, 1.6, 7.5, 2.2, 7.5)

    # Runtime
    draw_block(ax, 3.5, 7.5, 3.4, 2.0, 'Phase-Aware Runtime\nInstrumentation\n(llama.cpp)', facecolor='#FAFAFA', lw=1.5, fontsize=12)

    # Telemetry Pills
    draw_elbow_arrow(ax, 5.2, 7.5, 5.5, 8.2, 5.6, 8.2)
    draw_pill(ax, 6.6, 8.2, 2.2, 0.8, 'Phase Signal ($\phi$)')
    
    draw_elbow_arrow(ax, 5.2, 7.5, 5.5, 6.8, 5.6, 6.8)
    draw_pill(ax, 6.6, 6.8, 2.2, 0.8, 'Telemetry ($T, P$)')

    # Digital Twin
    draw_block(ax, 10.3, 6.8, 4.4, 2.4, 'Physics-Informed Digital Twin\n(PINN + Kalman Refinement)\n[10s Thermal Forecast Horizon]', facecolor='#FAFAFA', lw=1.5, fontsize=12)
    
    # Telemetry -> Twin
    draw_arrow(ax, 7.7, 6.8, 8.1, 6.8) 
    
    # Phase -> Twin
    draw_elbow_arrow(ax, 7.7, 8.2, 7.9, 7.5, 8.1, 7.5)

    # Offline Block
    draw_block(ax, 11.5, 10.8, 8.6, 1.8, '', facecolor=bg_offline, edgecolor='#999999', style="square,pad=0", lw=1.5, ls='--', zorder=0)
    ax.text(7.5, 11.4, 'Offline Calibration Phase', ha='left', va='center', fontsize=12, fontweight='bold')
    
    draw_pill(ax, 8.5, 10.5, 2.0, 0.8, 'Dataset ($\mathcal{D}_{\mathrm{thermo}}$)')
    draw_pill(ax, 11.5, 10.5, 3.4, 0.8, 'Thermal Parameter Identification\n($R_{\mathrm{th}}, C_{\mathrm{th}}$)', fontsize=10)
    draw_pill(ax, 14.5, 10.5, 2.0, 0.8, 'PINN Training')
    
    draw_arrow(ax, 9.5, 10.5, 9.8, 10.5)
    draw_arrow(ax, 13.2, 10.5, 13.5, 10.5)
    
    draw_elbow_arrow(ax, 14.5, 10.1, 14.5, 8.0, 12.5, 8.0)
    ax.text(13.75, 8.1, 'Weights', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # SYSTEM 2
    draw_block(ax, 7.5, 2.4, 14.5, 4.6, '', facecolor=bg_system, edgecolor='#888888', style="square,pad=0", lw=1.5, zorder=0)
    ax.text(0.5, 4.4, 'System 2: Safety-Constrained Scheduling Loop', ha='left', va='center', fontsize=14, fontweight='bold')

    # State Vector
    draw_pill(ax, 6.6, 2.8, 3.4, 1.2, 'State Vector\n$\mathbf{s}_t = (T, P, f, \phi, \mathbf{\hat{T}})$')
    
    # Arrows to State Vector
    draw_arrow(ax, 6.6, 7.8, 6.6, 3.3) # phi down
    
    # Twin -> State vector (hat T)
    draw_elbow_arrow(ax, 10.3, 5.8, 10.3, 4.8, 6.6, 4.8) # Down to 4.8, left to 6.6
    draw_arrow(ax, 6.6, 4.8, 6.6, 3.4)
    ax.text(8.45, 4.9, 'Predicted Temperature ($\mathbf{\hat{T}}$)', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # DQN
    draw_block(ax, 9.8, 2.8, 2.6, 2.0, 'D-DQN\nScheduler', facecolor='#FAFAFA', lw=1.5, fontsize=12)
    draw_arrow(ax, 7.8, 2.8, 8.5, 2.8) # state -> DQN

    # Shield
    draw_block(ax, 13.2, 2.8, 2.8, 2.0, 'Deterministic\nSafety Shield\n(Veto + Action Repair)', facecolor='#FAFAFA', lw=1.5, fontsize=12)
    draw_arrow(ax, 11.1, 2.8, 11.8, 2.8, text='$f_{\mathrm{cand}}$')

    # Twin straight to Shield (validation)
    draw_elbow_arrow(ax, 11.8, 5.8, 11.8, 4.8, 13.2, 4.8)
    draw_arrow(ax, 13.2, 4.8, 13.2, 3.8)
    ax.text(12.5, 4.9, 'Predicted Temperature ($\mathbf{\hat{T}}$)', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Action output
    draw_block(ax, 13.2, 0.5, 4.4, 0.9, 'Hardware Resource Controller', facecolor='white', lw=1.5, fontsize=12)
    draw_arrow(ax, 13.2, 1.8, 13.2, 0.95, text='Safe Action ($f^*$)')

    path = os.path.join(OUTPUT_DIR, 'fig2_architecture.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    parent_path = os.path.join(os.path.dirname(OUTPUT_DIR), 'fig2_architecture.png')
    plt.savefig(parent_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"-> Saved: {path} and {parent_path}")

if __name__ == '__main__':
    generate_architecture_diagram()
