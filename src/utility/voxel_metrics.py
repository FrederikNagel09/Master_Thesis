import torch


def compute_voxel_iou(
    a: torch.Tensor,
    b: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Compute pairwise volumetric IoU between two sets of voxel grids.

    Args:
        a         (torch.Tensor): generated samples, (N, 1, D, H, W), float in [0, 1]
        b         (torch.Tensor): reference shapes,  (M, 1, D, H, W), float in [0, 1]
        threshold (float):        binarisation threshold
    Returns:
        torch.Tensor: IoU distance matrix (N, M), float — 0 = identical, 1 = no overlap
    """
    # Binarise: (N, V) and (M, V) where V = D*H*W
    a_bin = (a >= threshold).float().view(a.shape[0], -1)
    b_bin = (b >= threshold).float().view(b.shape[0], -1)

    # Pairwise intersection and union via broadcasting: (N, M)
    intersection = torch.einsum("nv,mv->nm", a_bin, b_bin)
    a_sum = a_bin.sum(dim=1, keepdim=True)  # (N, 1)
    b_sum = b_bin.sum(dim=1, keepdim=True)  # (M, 1)
    union = a_sum + b_sum.T - intersection  # (N, M)

    iou = intersection / union.clamp(min=1e-8)
    return 1.0 - iou  # convert similarity → distance


def compute_mmd_cov(
    generated: torch.Tensor,
    reference: torch.Tensor,
    threshold: float = 0.5,
    batch_size: int = 64,
) -> tuple[float, float]:
    """
    Compute MMD and COV metrics between generated and reference voxel sets.

    MMD (Minimum Matching Distance): for each reference shape, find its closest
    generated shape and average those distances. Measures generation quality.

    COV (Coverage): fraction of reference shapes that are the nearest neighbour
    of at least one generated shape. Measures generation diversity.

    Args:
        generated  (torch.Tensor): model samples,     (N, 1, D, H, W), float in [0, 1]
        reference  (torch.Tensor): real val shapes,   (M, 1, D, H, W), float in [0, 1]
        threshold  (float):        IoU binarisation threshold
        batch_size (int):          row batch size for pairwise distance computation
    Returns:
        tuple: (mmd, cov)
            mmd (float): mean IoU distance from each reference to its nearest generated shape
            cov (float): fraction of reference shapes claimed by at least one generated shape
    """
    N = generated.shape[0]  # noqa: N806
    M = reference.shape[0]  # noqa: N806

    # Build full (N, M) distance matrix in row-batches to avoid OOM
    dist = torch.zeros(N, M, dtype=torch.float32)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        dist[start:end] = compute_voxel_iou(
            generated[start:end].cpu(),
            reference.cpu(),
            threshold=threshold,
        )

    # MMD: for each reference col, find min over generated rows → average
    mmd = dist.min(dim=0).values.mean().item()

    # COV: for each generated row, claim the nearest reference col;
    #      count how many distinct reference shapes got claimed
    nearest_ref = dist.argmin(dim=1)  # (N,) — which ref each gen sample claims
    cov = nearest_ref.unique().shape[0] / M

    return mmd, cov
