import torch
import torch.nn.functional as F

def pearson_loss(x, y, reduction='mean', eps=1e-8):
    """
    :param x:
    :param y:
    :return:

    Pearson correlation coefficient 是一种衡量两个变量之间线性关系的强度和方向的方法。
    其值范围在 -1 到 1 之间，1 表示完全的正相关，-1 表示完全的负相关，0 表示没有线性关系。
    pearson_distance 是 1-coefficient。
    # 示例
    x = torch.randn(32, 10)
    y = torch.randn(32, 10)
    distance = pearson_distance(x, y)
    print("Pearson distance:", distance)

    """
    assert x.shape == y.shape, "x and y must have the same shape"

    # 计算 x 和 y 的均值
    mean_x = torch.mean(x, dim=1, keepdim=True)
    mean_y = torch.mean(y, dim=1, keepdim=True)

    # 计算 x 和 y 的中心化版本
    vx = x - mean_x
    vy = y - mean_y

    # 计算 vx 和 vy 的点积、长度（L2范数）
    dot_product = torch.sum(vx * vy, dim=1)
    norm_x = torch.sqrt(torch.sum(vx ** 2, dim=1)+eps)
    norm_y = torch.sqrt(torch.sum(vy ** 2, dim=1)+eps)

    # 计算 Pearson correlation coefficient
    pearson_correlation = dot_product / (norm_x * norm_y+eps)

    # 计算 Pearson distance
    pearson_distance = 1 - pearson_correlation

    if reduction == 'mean':
        pearson_distance = pearson_distance.mean()

    return pearson_distance


def huber_loss(input, target, delta=1.0, reduction='mean'):
    """

    :param input:
    :param target:
    :param delta:
    :return:

    Huber loss 是一种在回归任务中常用的损失函数，它在损失值较小的时候表现为平方损失，而在损失值较大的时候表现为线性损失。
    这样可以减少对异常值的敏感度。以下是一个使用 PyTorch 实现 element-wise Huber loss 的例子：
    # 示例
    x = torch.randn(32, 10)
    y = torch.randn(32, 10)

    loss = huber_loss(x, y)
    print("Element-wise Huber loss:", loss)
    """
    assert input.shape == target.shape, "input and target must have the same shape"

    # 计算 input 和 target 之间的差值
    diff = input - target

    # 计算 element-wise Huber loss
    loss = torch.where(torch.abs(diff) < delta,
                       0.5 * diff ** 2,
                       delta * (torch.abs(diff) - 0.5 * delta))
    if reduction == 'mean':
        loss = loss.mean()


    return loss

def kl_loss(input, target, reduction='batchmean'):
    # 先转化为概率，之后取对数
    x_log = F.log_softmax(input, dim=-1)
    # 只转化为概率
    y = F.softmax(target, dim=-1)
    out = F.kl_div(x_log, y, size_average=None, reduce=None, reduction=reduction, log_target=False)

    return out