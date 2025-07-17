# 将数据进行检查，none、‘’， ‘ ’等转化为0
def process_key(key, default_value=0.0):
    """
    处理FDV数据，将其安全转换为浮点数

    参数:
        fdv: 待处理的FDV数据（可能是字符串、数字、None等）
        default_value: 转换失败时返回的默认值

    返回:
        转换后的浮点数或默认值
    """
    # 情况1: 如果是None，直接返回默认值
    if key is None:
        return default_value

    # 情况2: 如果已经是数字类型，直接返回
    if isinstance(key, (int, float)):
        # 排除布尔值（因为bool是int的子类）
        if isinstance(key, bool):
            return default_value
        return float(key)

    # 情况3: 如果是字符串类型，进行处理
    if isinstance(key, str):
        # 去除前后空白字符（包括空格、制表符、换行符等）
        stripped = key.strip()
        # 空字符串处理
        if not stripped:
            return default_value

        # 尝试转换（支持科学计数法）
        try:
            return float(stripped)
        except ValueError:
            return default_value

    # 情况4: 其他不支持的类型（如列表、字典等）
    return default_value
