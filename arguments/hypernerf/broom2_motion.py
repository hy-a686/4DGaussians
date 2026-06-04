_base_ = "./broom2.py"

OptimizationParams = dict(
    motion_loss_weight=1.0,
    motion_loss_quantile=0.95,
    motion_loss_min_response=0.0,
    motion_loss_group_by_stream=False,
)
