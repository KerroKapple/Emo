import torch

from src.optimize_distill import DistillationLoss


def test_distillation_loss_is_scalar_and_differentiable():
    crit = DistillationLoss(temperature=2.0, alpha=0.5)
    student = torch.randn(4, 5, requires_grad=True)
    teacher = torch.randn(4, 5)
    labels = torch.tensor([0, 1, 2, 3])

    total, hard, soft = crit(student, teacher, labels)
    assert total.dim() == 0
    assert torch.isfinite(total)
    assert hard.item() >= 0 and soft.item() >= 0

    total.backward()
    assert student.grad is not None


def test_distillation_alpha_weights_components():
    student = torch.randn(8, 5)
    teacher = torch.randn(8, 5)
    labels = torch.randint(0, 5, (8,))

    only_hard = DistillationLoss(alpha=1.0)(student, teacher, labels)[0]
    crit = DistillationLoss(alpha=1.0)
    hard = crit.ce(student, labels)
    assert torch.allclose(only_hard, hard, atol=1e-5)
