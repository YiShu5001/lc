import sys
import os

# 将项目根目录加入路径，以保证包导入正常
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Trainer.HGCTrainer import HGCTrainer

def main():
    print("==================================================")
    print("      Testing HGC-RL Pipeline Integration         ")
    print("==================================================")
    
    # 为了快速测试，我们设置极小的训练参数
    # 纯随机探索 50 步后，网络开始更新。
    # 每个 episode 最大 200 步。测试 3 个 episode 即退出。
    trainer = HGCTrainer(
        env_kwargs={'ctrl_freq': 100, 'rl_freq': 10, 'gui': False},
        max_episodes=3,
        max_steps_per_ep=200,
        batch_size=32,
        start_timesteps=50,
        explore_noise=0.1,
        model_save_path="./models/test_hgc_rl"
    )
    
    trainer.train()

if __name__ == "__main__":
    main()