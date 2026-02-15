"""
全局日志处理模块（使用 loguru）
支持控制台和文件输出，不同日志级别使用不同颜色和格式
"""
import sys
from pathlib import Path
from loguru import logger as module_logger
from typing import Optional


class LoggerManager:
    """日志管理器"""
    
    _initialized = False
    _current_config = {}
    
    # CRITICAL 级别的特殊格式
    critical_format = (
        "<bold><red>{time:YYYY-MM-DD HH:mm:ss.SSS}</red></bold> | "
        "<bold><red>{level: <8}</red></bold> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<bold><red>{message}</red></bold>"
    )
    
    # 普通日志格式
    normal_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    # 文件日志格式（无颜色）
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )
    
    @classmethod
    def setup(
        cls,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        max_bytes: int = 10485760,  # 10MB
        backup_count: int = 5,
        console_output: bool = True
    ):
        """
        配置日志系统
        
        Args:
            log_level: 日志级别 (TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL)
            log_file: 日志文件路径
            max_bytes: 单个日志文件最大字节数
            backup_count: 保留的日志文件数量
            console_output: 是否输出到控制台
        """
        # 保存配置
        cls._current_config = {
            'log_level': log_level,
            'log_file': log_file,
            'max_bytes': max_bytes,
            'backup_count': backup_count,
            'console_output': console_output
        }
        
        # 如果已经初始化过，先清除
        if cls._initialized:
            module_logger.remove()
        else:
            # 首次初始化，移除默认 logger
            module_logger.remove()
        
        # 自定义日志级别颜色
        module_logger.level("TRACE", color="<dim>")
        module_logger.level("DEBUG", color="<white>")
        module_logger.level("INFO", color="<cyan>")
        module_logger.level("SUCCESS", color="<bold><green>")
        module_logger.level("WARNING", color="<yellow>")
        module_logger.level("ERROR", color="<red>")
        module_logger.level("CRITICAL", color="<bold><red>")
        
        # 添加控制台输出
        if console_output:
            # CRITICAL 级别使用特殊格式（红色加粗）
            module_logger.add(
                sys.stdout,
                level="CRITICAL",
                colorize=True,
                format=cls.critical_format,
                filter=lambda record: record["level"].name == "CRITICAL",
                backtrace=True,
                diagnose=True
            )
            
            # 其他级别使用普通格式
            module_logger.add(
                sys.stdout,
                level=log_level,
                colorize=True,
                format=cls.normal_format,
                filter=lambda record: record["level"].name != "CRITICAL",
                backtrace=True,
                diagnose=True
            )
        
        # 添加文件输出（不带颜色）
        if log_file:
            # 确保日志目录存在
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 启动时删除旧的日志文件
            if log_path.exists():
                try:
                    log_path.unlink()
                    print(f"✓ 已删除旧日志文件: {log_file}")
                except Exception as e:
                    print(f"⚠ 删除日志文件失败: {e}")
            
            # 删除旧的压缩日志文件
            for old_log in log_path.parent.glob(f"{log_path.stem}*.zip"):
                try:
                    old_log.unlink()
                    print(f"✓ 已删除旧压缩日志: {old_log.name}")
                except Exception as e:
                    print(f"⚠ 删除压缩日志失败: {e}")
            
            # 文件日志 - 可选择是否包含颜色代码
            # 如果需要在文件中也保留颜色，设置 colorize=True 并使用 normal_format
            # 默认使用无颜色的 file_format，适合普通文本编辑器查看
            module_logger.add(
                log_file,
                level=log_level,
                format=cls.normal_format,  # 使用带颜色的格式
                colorize=True,  # 启用颜色代码
                rotation=max_bytes,  # 文件大小轮转
                retention=backup_count,  # 保留文件数量
                compression="zip",  # 压缩旧日志
                encoding="utf-8",
                backtrace=True,
                diagnose=True
            )
        
        cls._initialized = True
    
    @classmethod
    def get_config(cls):
        """获取当前配置"""
        return cls._current_config.copy()


def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10485760,
    backup_count: int = 5,
    console_output: bool = True
):
    """
    配置日志系统（便捷函数）
    
    Args:
        log_level: 日志级别 (TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径
        max_bytes: 单个日志文件最大字节数
        backup_count: 保留的日志文件数量
        console_output: 是否输出到控制台
    
    Example:
        >>> from util.logger import setup_logger
        >>> setup_logger(log_level="DEBUG", log_file="logs/app.log")
    """
    LoggerManager.setup(
        log_level=log_level,
        log_file=log_file,
        max_bytes=max_bytes,
        backup_count=backup_count,
        console_output=console_output
    )


def get_logger(name: str = None):
    """
    获取 logger 实例
    
    Args:
        name: logger 名称（用于标识日志来源）
    
    Returns:
        loguru.Logger 实例

    
    日志级别和颜色：
        - TRACE: 灰色 - 最详细的跟踪信息
        - DEBUG: 白色 - 调试信息
        - INFO: 青色 - 一般信息
        - SUCCESS: 绿色加粗 - 成功信息（loguru特有）
        - WARNING: 黄色 - 警告信息
        - ERROR: 红色 - 错误信息
        - CRITICAL: 红色加粗 - 严重错误（使用特殊格式）
    """
    # 如果还没有初始化，使用默认配置初始化
    if not LoggerManager._initialized:
        # 尝试从配置加载
        try:
            from config import get_config
            config = get_config()
            LoggerManager.setup(
                log_level=config.LOG_LEVEL,
                log_file=config.LOG_FILE,
                max_bytes=config.LOG_MAX_BYTES,
                backup_count=config.LOG_BACKUP_COUNT,
                console_output=True
            )
        except Exception:
            # 如果加载配置失败，使用默认配置
            LoggerManager.setup(
                log_level="INFO",
                log_file="logs/run.log",
                console_output=True
            )
    
    # 返回绑定了名称的 logger
    if name:
        return module_logger.bind(name=name)
    return module_logger


# 自动初始化默认 logger
logger = get_logger("app")

# 导出主要接口
__all__ = [
    'logger',
    'get_logger',
    'setup_logger',
    'LoggerManager'
]
