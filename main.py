"""
主训练入口
统一的训练脚本，支持命令行参数配置
"""
import argparse
import torch
import torch.optim as optim

from Gym_env.BaseEnv import EnvConfig
from Gym_env.factories.PyBulletDronesFactory import PyBulletDronesFactory
from NN.BaseNN import ModelConfig
from NN.model_factory import create_model_from_env
from Reinforce_learning.Basealgos import AlgoConfig
from Reinforce_learning.algo_factory import create_algo, is_on_policy
from Trainer.BaseTrainer import Trainer, TrainConfig
from Trainer.OffPolicyTrainer import OffPolicyTrainer
from Trainer.callbacks import CallbackList, ModelCheckpointCallback, LoggerCallback
from configs.default_configs import get_default_config


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="强化学习训练脚本")
    
    # 算法选择
    parser.add_argument("--algo", type=str, default="ppo",
                       choices=["ppo", "a2c", "sac", "td3", "ddpg", "dqn"],
                       help="算法名称")
    
    # 环境配置
    parser.add_argument("--env-id", type=str, default="HoverAviary-v0",
                       help="环境ID")
    parser.add_argument("--num-envs", type=int, default=1,
                       help="并行环境数")
    parser.add_argument("--num-drones", type=int, default=1,
                       help="无人机数量（MultiHoverAviary）")
    
    # 训练配置
    parser.add_argument("--total-timesteps", type=int, default=1_000_000,
                       help="总训练步数")
    parser.add_argument("--num-steps", type=int, default=128,
                       help="Rollout长度（on-policy）")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                       help="学习率")
    parser.add_argument("--device", type=str, default="cpu",
                       choices=["cpu", "cuda"],
                       help="设备")
    
    # 输出配置
    parser.add_argument("--output-dir", type=str, default="./results",
                       help="输出目录")
    parser.add_argument("--save-freq", type=int, default=10000,
                       help="模型保存频率")
    
    # 其他
    parser.add_argument("--seed", type=int, default=0,
                       help="随机种子")
    parser.add_argument("--gui", action="store_true",
                       help="显示GUI")
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    
    # 创建环境
    env_cfg = EnvConfig(
        env_id=args.env_id,
        num_envs=args.num_envs,
        seed=args.seed
    )
    env_cfg.gui = args.gui
    env_cfg.num_drones = args.num_drones
    
    env_factory = PyBulletDronesFactory(env_cfg)
    envs = env_factory.build()
    
    print(f"[INFO] 环境创建成功: {envs.obs_shape}, {envs.action_shape}, discrete={envs.is_discrete}")
    
    # 创建模型
    model_cfg = ModelConfig(
        hidden_sizes=(128, 128),
        activation="tanh"
    )
    model = create_model_from_env(
        cfg=model_cfg,
        obs_shape=envs.obs_shape,
        action_shape=envs.action_shape,
        is_discrete=envs.is_discrete
    )
    
    print(f"[INFO] 模型创建成功: {type(model).__name__}")
    
    # 创建算法
    from Reinforce_learning.algo_factory import get_algo_config_class
    algo_config_class = get_algo_config_class(args.algo)
    algo_cfg = algo_config_class(learning_rate=args.learning_rate)
    algo = create_algo(args.algo, algo_cfg)
    
    print(f"[INFO] 算法创建成功: {args.algo}")
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 创建训练配置
    train_cfg = TrainConfig(
        total_timesteps=args.total_timesteps,
        num_steps=args.num_steps,
        device=args.device
    )
    
    # 创建回调
    callbacks = CallbackList([
        ModelCheckpointCallback(
            save_path=args.output_dir,
            save_freq=args.save_freq
        ),
        LoggerCallback(log_interval=1000)
    ])
    callbacks.set_model(model)
    
    # 创建训练器
    if is_on_policy(args.algo):
        trainer = Trainer(
            envs=envs,
            model=model,
            algo=algo,
            optimizer=optimizer,
            cfg=train_cfg
        )
    else:
        trainer = OffPolicyTrainer(
            envs=envs,
            model=model,
            algo=algo,
            optimizer=optimizer,
            cfg=train_cfg
        )
    
    print(f"[INFO] 开始训练...")
    print(f"[INFO] 算法: {args.algo}, 环境: {args.env_id}, 总步数: {args.total_timesteps}")
    
    # 开始训练
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n[INFO] 训练被用户中断")
    
    print("[INFO] 训练完成")


if __name__ == "__main__":
    main()
