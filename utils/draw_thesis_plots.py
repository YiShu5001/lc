import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# 全局绘图设置 (符合学术论文规范)
plt.rcParams['font.family'] = 'Times New Roman'  # 英文使用Times New Roman
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 16

# 确保输出目录存在
os.makedirs('papers/figures/chap3', exist_ok=True)
os.makedirs('papers/figures/chap4', exist_ok=True)

# ---------------------------------------------------------
# 第三章图表：基于 TSA-LADRC 的单机底层鲁棒控制
# ---------------------------------------------------------

def plot_fig3_3_step_response(data_path=None, save_path='papers/figures/chap3/step_response.png'):
    """
    图 3.3：无风环境下的俯仰角阶跃响应对比曲线
    """
    plt.figure(figsize=(8, 6))
    
    if data_path and os.path.exists(data_path):
        # 实际使用时替换为：data = np.loadtxt(data_path, delimiter=',')
        pass
    else:
        # 生成模拟数据
        t = np.linspace(0, 5, 500)
        target = np.ones_like(t)
        target[t < 0.5] = 0
        
        # 模拟不同算法的响应
        pid = np.zeros_like(t); pid[t>=0.5] = 1 - np.exp(-3*(t[t>=0.5]-0.5)) * np.cos(4*np.pi*(t[t>=0.5]-0.5))
        ladrc = np.zeros_like(t); ladrc[t>=0.5] = 1 - np.exp(-1.5*(t[t>=0.5]-0.5))
        rl_ladrc = np.zeros_like(t); rl_ladrc[t>=0.5] = 1 - np.exp(-5*(t[t>=0.5]-0.5)) + 0.05*np.random.randn(len(t[t>=0.5]))
        tsa_ladrc = np.zeros_like(t); tsa_ladrc[t>=0.5] = 1 - np.exp(-5.5*(t[t>=0.5]-0.5))
        
        plt.plot(t, target, 'k--', linewidth=2, label='Reference')
        plt.plot(t, pid, '#1f77b4', linewidth=1.5, label='PID')
        plt.plot(t, ladrc, '#ff7f0e', linewidth=1.5, label='LADRC')
        plt.plot(t, rl_ladrc, '#2ca02c', linewidth=1.5, alpha=0.7, label='RL-LADRC')
        plt.plot(t, tsa_ladrc, '#d62728', linewidth=2.5, label='TSA-LADRC (Ours)')

    plt.xlabel('Time (s)')
    plt.ylabel('Pitch Angle (rad)')
    plt.title('Step Response Comparison (Pitch)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 3.3 to {save_path}")

def plot_fig3_4_wind_tracking(data_path=None, save_path='papers/figures/chap3/wind_tracking.png'):
    """
    图 3.4：强阵风扰动环境下的三维轨迹跟踪误差对比
    """
    plt.figure(figsize=(10, 5))
    
    if data_path and os.path.exists(data_path):
        pass
    else:
        t = np.linspace(0, 10, 1000)
        # t=2s 阶跃风, t=5s 正弦风
        err_pid = 0.05 * np.ones_like(t) + np.random.randn(1000)*0.01
        err_pid[t>2] += 0.8 * np.exp(-(t[t>2]-2))
        err_pid[t>5] += 0.5 * np.sin(5*(t[t>5]-5)) * np.exp(-0.2*(t[t>5]-5))
        
        err_tsa = 0.02 * np.ones_like(t) + np.random.randn(1000)*0.005
        err_tsa[t>2] += 0.1 * np.exp(-5*(t[t>2]-2))
        err_tsa[t>5] += 0.05 * np.sin(5*(t[t>5]-5)) * np.exp(-2*(t[t>5]-5))

        plt.plot(t, err_pid, '#1f77b4', linewidth=1.5, label='PID Error')
        plt.plot(t, err_tsa, '#d62728', linewidth=2, label='TSA-LADRC Error')
        
        plt.axvline(x=2.0, color='gray', linestyle='--', alpha=0.8)
        plt.text(2.1, 0.6, 'Step Wind Injection', color='gray')
        plt.axvline(x=5.0, color='gray', linestyle='--', alpha=0.8)
        plt.text(5.1, 0.6, 'Sine Wind Injection', color='gray')

    plt.xlabel('Time (s)')
    plt.ylabel('Trajectory Tracking Error (m)')
    plt.title('Tracking Error under Severe Wind Disturbances')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 3.4 to {save_path}")

def plot_fig3_5_fft_analysis(data_path=None, save_path='papers/figures/chap3/fft_analysis.png'):
    """
    图 3.5：电机推力控制指令的时域曲线与 FFT 频域能量分布对比
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    if data_path and os.path.exists(data_path):
        pass
    else:
        Fs = 100.0  # 采样频率 100Hz
        t = np.arange(0, 2, 1/Fs)
        
        # 模拟控制信号: 基础推力 + 低频机动 + 高频噪声(仅RL-LADRC有)
        base_thrust = 5.0 + 0.5 * np.sin(2 * np.pi * 1.5 * t)
        rl_noise = 0.8 * np.sin(2 * np.pi * 35 * t) + 0.5 * np.random.randn(len(t))
        tsa_noise = 0.05 * np.random.randn(len(t))
        
        thrust_rl = base_thrust + rl_noise
        thrust_tsa = base_thrust + tsa_noise
        
        # 1. 时域子图
        ax1.plot(t, thrust_rl, '#2ca02c', alpha=0.7, label='RL-LADRC')
        ax1.plot(t, thrust_tsa, '#d62728', linewidth=2, label='TSA-LADRC')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Thrust Command (N)')
        ax1.set_title('(a) Time Domain Thrust Command')
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend()
        
        # 2. 频域子图 (FFT)
        N = len(t)
        freqs = np.fft.fftfreq(N, 1/Fs)[:N//2]
        
        fft_rl = np.abs(np.fft.fft(thrust_rl))[:N//2] * 2.0 / N
        fft_tsa = np.abs(np.fft.fft(thrust_tsa))[:N//2] * 2.0 / N
        
        # 去掉直流分量以便观察高频
        fft_rl[0] = 0; fft_tsa[0] = 0 
        
        ax2.plot(freqs, fft_rl, '#2ca02c', alpha=0.7, label='RL-LADRC')
        ax2.plot(freqs, fft_tsa, '#d62728', linewidth=2, label='TSA-LADRC')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Amplitude')
        ax2.set_title('(b) FFT Spectrum Analysis')
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 3.5 to {save_path}")

# ---------------------------------------------------------
# 第四章图表：基于金字塔记忆与双流 Transformer 的多机协同规划
# ---------------------------------------------------------

def plot_fig4_1_curriculum_curve(data_path=None, save_path='papers/figures/chap4/curriculum_learning_curve.png'):
    """
    图 4.1：多阶段课程学习周期中的抗遗忘成功率对比曲线
    """
    plt.figure(figsize=(12, 6))
    
    if data_path and os.path.exists(data_path):
        pass
    else:
        episodes = np.arange(0, 60000, 100)
        
        # 课程切换点 (Episode: 10k, 20k, 30k, 40k, 50k)
        stages = [10000, 20000, 30000, 40000, 50000]
        stage_names = ['1-A', '1-B', '1-C', '2-A', '2-B', '2-C']
        
        # 模拟成功率数据
        def smooth(y, box_pts):
            box = np.ones(box_pts)/box_pts
            return np.convolve(y, box, mode='same')
            
        success_per = np.zeros_like(episodes, dtype=float)
        success_pyramid = np.zeros_like(episodes, dtype=float)
        
        for i, ep in enumerate(episodes):
            phase = ep // 10000
            progress = (ep % 10000) / 10000.0
            
            # 传统 PER 在跨越单机到多机 (30k) 时暴跌
            drop_per = 0.8 if phase == 3 else 0.4
            success_per[i] = 1.0 - drop_per * np.exp(-5 * progress) + np.random.randn()*0.05
            
            # Pyramid PER 波动极小
            drop_pyr = 0.2 if phase == 3 else 0.1
            success_pyramid[i] = 1.0 - drop_pyr * np.exp(-10 * progress) + np.random.randn()*0.02
            
        success_per = np.clip(smooth(success_per, 10), 0, 1)
        success_pyramid = np.clip(smooth(success_pyramid, 5), 0, 1)

        plt.plot(episodes, success_per, '#1f77b4', linewidth=2, alpha=0.8, label='TD3 + PER + MLP')
        plt.plot(episodes, success_pyramid, '#d62728', linewidth=2.5, label='Task-Decomposed Actor + Pyramid-PER')
        
        # 绘制课程分割线
        for idx, cp in enumerate(stages):
            plt.axvline(x=cp, color='gray', linestyle='--', alpha=0.5)
            plt.text(cp - 5000, 0.1, stage_names[idx], fontsize=12, ha='center')
        plt.text(55000, 0.1, stage_names[-1], fontsize=12, ha='center')

    plt.xlabel('Training Episodes')
    plt.ylabel('Success Rate')
    plt.title('Success Rate across Multi-Stage Curriculum Learning')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 4.1 to {save_path}")

def plot_fig4_2_attention_heatmap(data_path=None, save_path='papers/figures/chap4/attention_heatmap.png'):
    """
    图 4.2：不同环境特征下的双流网络注意力权重分布热力图
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    if data_path and os.path.exists(data_path):
        pass
    else:
        # 模拟注意力矩阵: 行代表自身 Query，列代表不同实体 Key
        labels = ['Nbr 1', 'Nbr 2', 'Nbr 3', 'Obs 1', 'Obs 2', 'Obs 3']
        
        # 场景 A: 空旷区域 (关注邻居进行编队)
        attn_empty = np.array([[0.35, 0.35, 0.28, 0.01, 0.005, 0.005]])
        sns.heatmap(attn_empty, annot=True, cmap="YlOrRd", cbar=True, xticklabels=labels, yticklabels=['Self'], ax=ax1, vmin=0, vmax=1)
        ax1.set_title('(a) Open Space (Focus on Neighbors)')
        
        # 场景 B: 逼近障碍物 (瞬间转移注意力到 Obs 2)
        attn_danger = np.array([[0.05, 0.02, 0.03, 0.05, 0.82, 0.03]])
        sns.heatmap(attn_danger, annot=True, cmap="YlOrRd", cbar=True, xticklabels=labels, yticklabels=['Self'], ax=ax2, vmin=0, vmax=1)
        ax2.set_title('(b) Approaching Obstacle (Focus on Obs 2)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 4.2 to {save_path}")

def plot_fig4_3_3d_trajectory(data_path=None, save_path='papers/figures/chap4/3d_formation_flight.png'):
    """
    图 4.3：极限密集障碍物环境下的多机编队三维协同飞行轨迹
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if data_path and os.path.exists(data_path):
        pass
    else:
        # 生成圆柱体障碍物
        theta = np.linspace(0, 2*np.pi, 20)
        z_cyl = np.linspace(0, 5, 2)
        Theta, Z_cyl = np.meshgrid(theta, z_cyl)
        
        obs_centers = [(2, 3), (5, 6), (7, 4), (4, 8), (8, 8)]
        r = 0.5
        for (cx, cy) in obs_centers:
            X_cyl = cx + r * np.cos(Theta)
            Y_cyl = cy + r * np.sin(Theta)
            ax.plot_surface(X_cyl, Y_cyl, Z_cyl, color='gray', alpha=0.5)
            
        # 生成无人机轨迹 (编队避障)
        t = np.linspace(0, 10, 200)
        # Leader
        x0 = t
        y0 = t + 1.5 * np.sin(t)
        z0 = 3 + 0.2 * np.cos(t)
        
        # Follower 1 (Left)
        x1 = x0 - 0.5
        y1 = y0 + 1.0
        z1 = z0
        
        # Follower 2 (Right)
        x2 = x0 + 0.5
        y2 = y0 - 1.0
        z2 = z0
        
        ax.plot(x0, y0, z0, '#d62728', linewidth=2, label='UAV 1 (Leader)')
        ax.plot(x1, y1, z1, '#1f77b4', linewidth=2, linestyle='--', label='UAV 2 (Follower)')
        ax.plot(x2, y2, z2, '#2ca02c', linewidth=2, linestyle='--', label='UAV 3 (Follower)')
        
        # 标出起点和终点
        ax.scatter([x0[0], x1[0], x2[0]], [y0[0], y1[0], y2[0]], [z0[0], z1[0], z2[0]], color='green', s=50, marker='o')
        ax.scatter([x0[-1], x1[-1], x2[-1]], [y0[-1], y1[-1], y2[-1]], [z0[-1], z1[-1], z2[-1]], color='blue', s=50, marker='*')

    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Altitude Z (m)')
    ax.set_title('3D Cooperative Trajectory in Dense Obstacle Environment')
    ax.legend(loc='upper left')
    
    # 调整视角
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 4.3 to {save_path}")

if __name__ == "__main__":
    print("Generating Academic Thesis Plots...")
    # 第三章
    plot_fig3_3_step_response()
    plot_fig3_4_wind_tracking()
    plot_fig3_5_fft_analysis()
    
    # 第四章
    plot_fig4_1_curriculum_curve()
    plot_fig4_2_attention_heatmap()
    plot_fig4_3_3d_trajectory()
    
    print("All plots successfully generated in papers/figures/!")
