import torch
import math

EPS = 1e-8  # to avoid log(0)


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """
    Compute KL divergence D_KL(p || q)
    p and q: tensors of shape (N,)
    """
    p = p + EPS
    q = q + EPS
    kl = torch.sum(p * torch.log(p / q))
    return kl.item()


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """
    Compute Jensen-Shannon divergence between p and q
    Symmetric and bounded between 0 and 1 (if log base 2)
    """
    p = p + EPS
    q = q + EPS
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * torch.log(p / m))
    kl_qm = torch.sum(q * torch.log(q / m))
    js = 0.5 * (kl_pm + kl_qm)
    return js.item()


def total_variation(p: torch.Tensor, q: torch.Tensor) -> float:
    """
    Compute Total Variation distance (L1 / 2)
    """
    tv = 0.5 * torch.sum(torch.abs(p - q))
    return tv.item()


# Optional: example usage
if __name__ == "__main__":
    p = torch.tensor([5.8174e-05, 1.6689e-05, 1.4722e-05, 1.8954e-05, 1.0])
    q = torch.tensor([6.1989e-05, 1.1399e-06, 4.2677e-05, 1.0058e-06, 1.0])

    print("KL(p || q):", kl_divergence(p, q))
    print("KL(q || p):", kl_divergence(q, p))
    print("JS(p || q):", js_divergence(p, q))
    print("Total Variation:", total_variation(p, q))