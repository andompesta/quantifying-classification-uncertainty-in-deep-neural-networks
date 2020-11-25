
from torch.utils.data import DataLoader
import argparse
import torch.optim as optim
import torch
from typing import *
from src.conf import LeNetConf
from src.model import LeNet, BaseModel, edl_mse_loss
from src.utils.dataset import get_mnist_dataset, mnist_collate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", default=50, type=int,
                        help="Desired number of epochs.")
    parser.add_argument("--max_grad_norm", default=1., type=float)

    return parser.parse_args()


def train_fn(
        dl: DataLoader,
        model: BaseModel,
        optim: optim.Optimizer,
        scheduler: optim.lr_scheduler.LambdaLR,
        device: torch.device,
        steps: int,
        epoch: int,
        args: argparse.Namespace
):
    total_loss = 0
    total_correct = 0
    evidence = 0
    num_predictions = 0
    model.train()

    def one_hot_embedding(
            labels: torch.Tensor,
            num_classes=model.conf.num_labels
    ) -> torch.Tensor:
        y = torch.eye(num_classes, device=labels.device)
        return y[labels]



    for i, (x, labels) in enumerate(dl):
        x = x.to(device)
        labels = labels.to(device)
        optim.zero_grad()

        with torch.set_grad_enabled(True):
            y = one_hot_embedding(labels)

            logits_t = model(x)

            loss_t, evidence, uncertanty, prob_t = edl_mse_loss(
                logits_t,
                y,
                epoch,
                model.conf.num_labels,
                annealing_step=10
            )
            loss_t = loss_t.mean()

            loss_t.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optim.step()
            if scheduler is not None:
                scheduler.step()
            steps += 1

        _, preds = torch.max(logits_t, 1)
        match = torch.eq(preds, labels).float()
        total_correct += match.sum().item()
        num_predictions += labels.size(0)

        # total_evidence = evidence.sum(dim=1, keepdims=True)
        # mean_evidence = total_evidence.mean()
        # mean_ev_succ = (total_evidence * match) / torch.sum(match + 1e-20)
        # mean_ev_fail = (total_evidence * (1 - match)) / (torch.sum(torch.abs(1 - match)) + 1e-20)

        total_loss += loss_t.item()

    total_loss /= i + 1
    accuracy = total_correct / num_predictions

    return steps, total_loss, accuracy


def get_group_params(
        named_parameters: List[Tuple[str, torch.nn.Parameter]],
        weight_decay: float,
        no_decay: Optional[List[str]] = None
):
    """
    package the parameters in 2 groups for proper weight decay
    :param named_parameters: named parameters list
    :param weight_decay: weight decay to use
    :param no_decay: list of parameter with no decay
    :return:
    """
    optimizer_grouped_parameters = [
        dict(
            params=[p for n, p in named_parameters if not any(nd in n for nd in no_decay)],
            weight_decay=weight_decay
        ), dict(
            params=[p for n, p in named_parameters if any(nd in n for nd in no_decay)],
            weight_decay=0.
        )
    ]
    return optimizer_grouped_parameters



if __name__ == '__main__':
    args = parse_args()
    device = torch.device(args.device)
    train, test = get_mnist_dataset()
    torch.manual_seed(0)

    conf = LeNetConf()
    model = LeNet(conf)
    model.init_weights()
    model = model.to(device)

    group_paramter = get_group_params(
        list(model.named_parameters()),
        weight_decay=0.005,
        no_decay=["conv1", "conv2", "fc1.bias", "fc2.bias"]
    )
    optimizer = optim.Adam(group_paramter, lr=3e-4)
    # exp_lr_scheduler = optim.lr_scheduler.StepLR(
    #     optimizer, step_size=7, gamma=0.1)

    train_dl = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=mnist_collate
    )

    test_dl = DataLoader(
        test,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=mnist_collate
    )


    global_step = 0
    losses = []
    train_accuracy = []

    for epoch in range(args.epochs):
        print(f"epoch {epoch}")

        global_step, loss, accuracy = train_fn(
            train_dl,
            model,
            optimizer,
            None,
            device,
            global_step,
            epoch,
            args
        )
        losses.append(loss)
        train_accuracy.append(accuracy)

        print(f"loss:{loss}\taccuracy:{accuracy}")

