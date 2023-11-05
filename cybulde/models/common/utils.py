import os

from contextlib import contextmanager
from typing import Generator

import torch


def get_local_rank() -> int:
    return int(os.getenv("LOCAL_RANK", -1))


def get_global_rank() -> int:
    return int(os.getenv("RANK", get_local_rank()))


@contextmanager
def local_rank_zero_first() -> Generator[None, None, None]:
    if not torch.distributed.is_initialized() and os.getenv("RANK") is not None:
        raise RuntimeError("RANK is set but torch.distributed is not initialized")

    if torch.distributed.is_initialized():
        rank = get_local_rank()
        if rank not in [-1, 0]:
            torch.distributed.barrier()  # type: ignore
        yield
        if rank == 0:
            torch.distributed.barrier()  # type: ignore
    else:
        yield


@contextmanager
def global_rank_zero_first() -> Generator[None, None, None]:
    if not torch.distributed.is_initialized() and os.getenv("RANK") is not None:
        raise RuntimeError("RANK is set but torch.distributed is not initialized")

    if torch.distributed.is_initialized():
        rank = get_global_rank()
        if rank not in [-1, 0]:
            torch.distributed.barrier()  # type: ignore
        yield
        if rank == 0:
            torch.distributed.barrier()  # type: ignore
    else:
        yield
