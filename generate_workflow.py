# pyrefly: ignore [missing-import]
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

def draw_diamond(ax, x, y, w, h, text, facecolor='white', edgecolor='black', text_color='black', lw=1.5, zorder=2, fontsize=10):
    # Diamond is a polygon
    pts = [
        [x, y + h/2],  # Top
        [x + w/2, y],  # Right
        [x, y - h/2],  # Bottom
        [x - w/2, y]   # Left
    ]
    poly = patches.Polygon(pts, closed=True, ec=edgecolor, fc=facecolor, lw=lw, zorder=zorder)
    ax.add_patch(poly)
    if text:
        ax.text(x, y, text, ha='center', va='center', 
                color=text_color, fontsize=fontsize, fontweight='bold', family='sans-serif', zorder=zorder+1)

def draw_arrow(ax, start_x, start_y, end_x, end_y, text=None, text_offset_x=0.0, text_offset_y=0.0, ha='center', va='bottom', fontsize=9):
    ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                arrowprops=dict(arrowstyle='-|>', lw=1.5, color='black', shrinkA=0, shrinkB=0),
                zorder=1)
    if text:
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        ax.text(mid_x + text_offset_x, mid_y + text_offset_y, text, ha=ha, va=va, fontsize=fontsize, fontweight='bold', family='sans-serif', zorder=4)

def draw_elbow_arrow(ax, start_x, start_y, mid_x, mid_y, end_x, end_y, text=None, text_offset_x=0.0, text_offset_y=0.0, ha='center', va='bottom', fontsize=9):
    ax.plot([start_x, mid_x], [start_y, mid_y], color='black', lw=1.5, zorder=1)
    draw_arrow(ax, mid_x, mid_y, end_x, end_y, text=text, text_offset_x=text_offset_x, text_offset_y=text_offset_y, ha=ha, va=va, fontsize=fontsize)

def generate_workflow_diagram():
    print("Generating Academic Pastel Workflow Diagram...")
    # Increased figure height to accommodate the new 10s forecast block
    fig, ax = plt.subplots(figsize=(10, 13.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13.5)
    ax.axis('off')

    # Define color palette (light/pastel)
    color_obs = '#E3F2FD'   # Light Blue for Observation
    color_fore = '#FFF3E0'  # Light Orange for Forecast
    color_opt = '#E8F5E9'   # Light Green for Optimization/Learning
    color_safe = '#FFEBEE'  # Light Red/Pink for Safety/Shield
    color_act = '#F3E5F5'   # Light Purple for Actuation/Execution

    # Title at the top
    ax.text(5.0, 13.0, 'End-to-End Operational Workflow of Thermo-LLM',
            ha='center', va='center', fontsize=12, fontweight='bold', family='sans-serif', zorder=5)

    # Draw Flow Blocks
    draw_block(ax, 5.0, 12.2, 3.0, 0.45, 'LLM Inference Request', facecolor=color_obs)
    draw_block(ax, 5.0, 11.3, 3.0, 0.45, r'Phase Detection ($\phi$)', facecolor=color_obs)
    draw_block(ax, 5.0, 10.4, 3.0, 0.45, 'Telemetry Collection', facecolor=color_obs)
    
    draw_block(ax, 5.0, 9.5, 3.5, 0.45, 'Physics-Informed Digital Twin', facecolor=color_fore)
    draw_block(ax, 5.0, 8.6, 3.5, 0.45, r'10 s Thermal Forecast ($\hat{T}$)', facecolor=color_fore)
    
    draw_block(ax, 5.0, 7.6, 3.8, 0.65, r'State Vector Construction' + '\n' + r'$s_t = (T, P, f, \phi, \hat{T})$', facecolor=color_opt)
    draw_block(ax, 5.0, 6.6, 3.8, 0.65, r'D-DQN Scheduler' + '\n' + r'Candidate Frequency ($f_{\mathrm{cand}}$)', facecolor=color_opt)
    
    # Diamond for verification (Safety Shield)
    draw_diamond(ax, 5.0, 5.2, 4.2, 1.2, r'Deterministic Safety Shield' + '\n' + r'$\max(\hat{T}) < T_{\mathrm{limit}}$', facecolor=color_safe, fontsize=9.5)
    
    # Branches (Safety / Repair and Execution)
    draw_block(ax, 8.4, 5.2, 2.2, 0.55, r'Action Repair' + '\n' + r'(Linear Scan)', facecolor=color_safe)
    draw_block(ax, 5.0, 3.8, 2.2, 0.5, r'Execute' + '\n' + r'Candidate Action', facecolor=color_act)
    
    draw_block(ax, 5.0, 2.8, 3.5, 0.55, r'Hardware Resource Controller' + '\n' + r'(DVFS Actuation)', facecolor=color_act)
    draw_block(ax, 5.0, 1.8, 3.5, 0.55, r'Reward Computation' + '\n' + r'(Throughput + Thermal Safety)', facecolor=color_opt)
    draw_block(ax, 5.0, 0.8, 3.0, 0.5, 'D-DQN Policy Update', facecolor=color_opt)

    # Connecting Arrows
    draw_arrow(ax, 5.0, 11.975, 5.0, 11.525)
    draw_arrow(ax, 5.0, 11.075, 5.0, 10.625)
    draw_arrow(ax, 5.0, 10.175, 5.0, 9.725)
    draw_arrow(ax, 5.0, 9.275, 5.0, 8.825)
    draw_arrow(ax, 5.0, 8.375, 5.0, 7.925)
    draw_arrow(ax, 5.0, 7.275, 5.0, 6.925)
    draw_arrow(ax, 5.0, 6.275, 5.0, 5.8)  # Connects to top point of diamond (y=5.2 + 0.6 = 5.8)
    
    # Safe branch (down)
    draw_arrow(ax, 5.0, 4.6, 5.0, 4.05, text='Safe', text_offset_x=0.15, text_offset_y=0.1, ha='left', va='center')
    # Unsafe branch (right)
    draw_arrow(ax, 7.1, 5.2, 7.3, 5.2, text='Unsafe', text_offset_x=0.0, text_offset_y=0.15, ha='center', va='bottom')
    
    # Action Repair connection down to Hardware Controller
    draw_elbow_arrow(ax, 8.4, 4.925, 8.4, 2.8, 6.75, 2.8)
    
    # Execute to Hardware Controller
    draw_arrow(ax, 5.0, 3.55, 5.0, 3.075)
    
    # Hardware Controller down to Reward
    draw_arrow(ax, 5.0, 2.525, 5.0, 2.075)
    # Reward down to Policy Update
    draw_arrow(ax, 5.0, 1.525, 5.0, 1.05)
    
    # Feedback loop back to D-DQN Action Selection
    draw_elbow_arrow(ax, 3.5, 0.8, 1.2, 0.8, 1.2, 6.6)
    draw_arrow(ax, 1.2, 6.6, 3.1, 6.6)
    ax.text(1.3, 3.7, 'Feedback Policy Update', ha='left', va='center', rotation=90, fontsize=10, fontweight='bold', family='sans-serif', zorder=4)

    path = os.path.join(OUTPUT_DIR, 'fig4_workflow.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    
    parent_path = os.path.join(os.path.dirname(OUTPUT_DIR), 'fig4_workflow.png')
    plt.savefig(parent_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"-> Saved: {path} and {parent_path}")

if __name__ == '__main__':
    generate_workflow_diagram()
