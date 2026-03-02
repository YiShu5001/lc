"""
训练回调函数
支持模型保存、日志记录、早停等功能
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Optional
from pathlib import Path

import torch


class BaseCallback(ABC):
    """
    回调基类
    """
    
    def __init__(self):
        self.model = None
        self.global_step = 0
        self.episode = 0
    
    def set_model(self, model):
        """设置模型"""
        self.model = model
    
    @abstractmethod
    def on_step(self, metrics: Dict[str, float]) -> bool:
        """
        每步调用
        
        Args:
            metrics: 训练指标
        
        Returns:
            True表示继续训练，False表示停止训练
        """
        pass
    
    def on_episode_end(self, episode_return: float, episode_length: int):
        """回合结束调用"""
        self.episode += 1
        pass


class ModelCheckpointCallback(BaseCallback):
    """
    模型保存回调
    """
    
    def __init__(
        self,
        save_path: str,
        save_freq: int = 10000,
        save_best: bool = True,
        best_metric: str = "episode_return",
        mode: str = "max"
    ):
        """
        Args:
            save_path: 保存路径
            save_freq: 保存频率（步数）
            save_best: 是否保存最佳模型
            best_metric: 用于判断最佳模型的指标
            mode: "max" 或 "min"
        """
        super().__init__()
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.save_freq = save_freq
        self.save_best = save_best
        self.best_metric = best_metric
        self.mode = mode
        self.best_value = float('-inf') if mode == "max" else float('inf')
        self.last_save_step = 0
    
    def on_step(self, metrics: Dict[str, float]) -> bool:
        """每步调用"""
        self.global_step = metrics.get("global_step", self.global_step)
        
        # 定期保存
        if self.global_step - self.last_save_step >= self.save_freq:
            self._save_model(f"checkpoint_{self.global_step}.pt")
            self.last_save_step = self.global_step
        
        # 保存最佳模型
        if self.save_best and self.best_metric in metrics:
            value = metrics[self.best_metric]
            if (self.mode == "max" and value > self.best_value) or \
               (self.mode == "min" and value < self.best_value):
                self.best_value = value
                self._save_model("best_model.pt")
        
        return True
    
    def _save_model(self, filename: str):
        """保存模型"""
        if self.model is not None:
            torch.save(self.model.state_dict(), self.save_path / filename)


class EarlyStoppingCallback(BaseCallback):
    """
    早停回调
    """
    
    def __init__(
        self,
        patience: int = 100,
        min_delta: float = 0.0,
        metric: str = "episode_return",
        mode: str = "max"
    ):
        """
        Args:
            patience: 容忍的步数（没有改善）
            min_delta: 最小改善幅度
            metric: 监控的指标
            mode: "max" 或 "min"
        """
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.mode = mode
        self.best_value = float('-inf') if mode == "max" else float('inf')
        self.wait = 0
    
    def on_step(self, metrics: Dict[str, float]) -> bool:
        """每步调用"""
        if self.metric not in metrics:
            return True
        
        value = metrics[self.metric]
        
        # 判断是否改善
        improved = False
        if self.mode == "max":
            if value > self.best_value + self.min_delta:
                improved = True
                self.best_value = value
        else:
            if value < self.best_value - self.min_delta:
                improved = True
                self.best_value = value
        
        if improved:
            self.wait = 0
        else:
            self.wait += 1
        
        # 判断是否早停
        if self.wait >= self.patience:
            return False
        
        return True


class LoggerCallback(BaseCallback):
    """
    日志记录回调
    """
    
    def __init__(self, log_interval: int = 100):
        """
        Args:
            log_interval: 日志记录间隔（步数）
        """
        super().__init__()
        self.log_interval = log_interval
        self.metrics_history = []
    
    def on_step(self, metrics: Dict[str, float]) -> bool:
        """每步调用"""
        self.global_step = metrics.get("global_step", self.global_step)
        
        if self.global_step % self.log_interval == 0:
            self._log(metrics)
        
        return True
    
    def _log(self, metrics: Dict[str, float]):
        """记录日志"""
        # 简单的print日志（可以扩展为TensorBoard/WandB）
        print(f"Step {self.global_step}: {metrics}")
        self.metrics_history.append(metrics.copy())


class CallbackList:
    """
    回调列表
    管理多个回调函数
    """
    
    def __init__(self, callbacks: list[BaseCallback]):
        """
        Args:
            callbacks: 回调函数列表
        """
        self.callbacks = callbacks
    
    def set_model(self, model):
        """设置模型"""
        for callback in self.callbacks:
            callback.set_model(model)
    
    def on_step(self, metrics: Dict[str, float]) -> bool:
        """每步调用所有回调"""
        for callback in self.callbacks:
            if not callback.on_step(metrics):
                return False  # 如果有回调返回False，停止训练
        return True
    
    def on_episode_end(self, episode_return: float, episode_length: int):
        """回合结束调用所有回调"""
        for callback in self.callbacks:
            callback.on_episode_end(episode_return, episode_length)
