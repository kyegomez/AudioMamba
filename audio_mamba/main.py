import torch
from torch import nn, Tensor
from zeta import SSM
from loguru import logger



class AuM(nn.Module):
    def __init__(
        self,
        dim: int,
        d_conv: int,
        dt_state: int = None,
        dim_inner: int = 128,
        dt_rank: int = 32,
    ):
        super().__init__()
        self.ssm = SSM(
            in_features=dim,
            dt_rank=dt_rank,
            dim_inner=dim_inner,
            d_state=dt_state,
        )

        # Proj
        self.proj = nn.Linear(dim, dim)

        # 1d conv
        self.conv = nn.Conv1d(
            in_channels=1000,
            out_channels=dim,
            kernel_size=d_conv,
            stride=1,
            padding=1,
            bias=False,
        )

        # SILU
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        b, s, d = x.shape
        logger.info(f"Input shape: {x.shape}")

        # Proj
        path1 = self.proj(x)
        path_2 = path1

        # Block
        path_2 = self.conv(path_2)
        logger.info(f"Path2 shape: {path_2.shape}")
        # Activation
        path_2 = self.act(path_2)
        # SSM
        path_2 = self.ssm(path_2)
        logger.info(f"SSM Shape: {path_2.shape}")

        # path1
        path1 = self.act(path1)

        # Projection
        mult = path1 * path_2

        return self.proj(mult)


x = torch.randn(1, 1000, 128)

model = AuM(
    dim=128,
    d_conv=3,
    dt_state=32,
    dim_inner=128,
)

# forward pass
out = model(x)
print(out.shape)
