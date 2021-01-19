import torch
from torch.nn import functional as F
from src.conf import BaseConf
from abc import abstractmethod

class BaseModel(torch.nn.Module):
    def __init__(self, conf: BaseConf):
        super(BaseModel, self).__init__()
        self.conf = conf
        self.name = conf.name

    def init_weights(self) -> None:
        """Initialize weights if needed."""
        self.apply(self._init_weights)

    @abstractmethod
    def _init_weights(self, module: torch.nn.Module):
        """Child model has to define the initialization policy."""
        ...
    def forward(self, *args, **kwargs):
        ...

    @classmethod
    def load(cls, conf: BaseConf, path_: str, mode: str = 'trained'):
        model = cls(conf)
        state_dict = torch.load(path_, map_location="cpu")

        if mode == 'pre-trained':
            strict = False
        elif mode == 'trained':
            strict = True
        else:
            raise NotImplementedError()

        model.load_state_dict(state_dict, strict=strict)
        return model


def relu_evidence(logits: torch.Tensor) -> torch.Tensor:
    return F.relu(logits)

def exp_evidence(logits: torch.Tensor) -> torch.Tensor:
    return torch.exp(torch.clamp(logits, -10, 10))

def softplus_evidence(logits: torch.Tensor) -> torch.Tensor:
    return F.softplus(logits)


def KL(alpha: torch.Tensor, num_classes):
    beta = torch.ones([1, num_classes], dtype=torch.float32, device=alpha.device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)

    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def mse_loss(
        alpha: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
        epoch: int,
        annealing_step: int
):
    S = torch.sum(alpha, dim=1, keepdim=True)
    m = alpha / S

    A = torch.sum((target - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    loglikelihood = A + B

    annealing_coef = min(1., epoch / annealing_step)

    kl_alpha = (alpha - 1) * (1 - target) + 1
    kl_div = annealing_coef * KL(kl_alpha, num_classes)

    return loglikelihood + kl_div

def one_hot_embedding(
        labels: torch.Tensor,
        num_classes: int
) -> torch.Tensor:
    y = torch.eye(num_classes, device=labels.device)
    return y[labels]

def edl_mse_loss(
        logits: torch.Tensor,
        target: torch.Tensor,
        epoch: int,
        num_classes: int,
        annealing_step: int,
):
    target = one_hot_embedding(target, num_classes)
    evidence = relu_evidence(logits)
    alpha = evidence + 1

    # uncertanty
    u = num_classes / torch.sum(alpha, dim=1, keepdim=True)
    # probability over class
    prob = alpha / torch.sum(alpha, dim=1, keepdim=True)

    loss = mse_loss(
        alpha,
        target,
        num_classes,
        epoch,
        annealing_step
    )

    return loss, evidence, u, prob