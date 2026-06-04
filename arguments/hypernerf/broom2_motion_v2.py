_base_ = "./broom2.py"

OptimizationParams = dict(
    motion_loss_weight=0.5,
    motion_loss_quantile=0.95,
    motion_loss_min_response=0.8,
    motion_loss_group_by_stream=True,
)