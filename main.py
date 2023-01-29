from datetime import datetime
from torch.utils.data import DataLoader
import argparse
import torch.optim as optim
import torch
from typing import *
from dynaconf import settings
from os import path
from src.conf import LeNetConf
from src.model import LeNet
from src.task import UncertaintyTask
from src.utils.dataset import get_mnist_dataset, mnist_collate
from src.utils import save_checkpoint
from src.utils.plot import create_timeseries, plot_scalars

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", default=50, type=int,
                        help="Desired number of epochs.")
    parser.add_argument("--max_grad_norm", default=1., type=float)
    parser.add_argument("--eval_every", default=5, type=int)

    return parser.parse_args()



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

    task = UncertaintyTask(
        "uncertainty-estimation"
    )
    exp_name = f'exp-{task.name}-MNIST-{model.name}-{datetime.now().strftime("%d-%m-%y_%H-%M-%S")}'

    best_f1 = 0.
    global_step = 0
    losses = []
    train_accuracy = []
    train_epochs = []

    eval_epochs = []
    eval_accuracy = []
    eval_precision = []
    eval_recall = []
    eval_f_score = []

    for epoch in range(args.epochs):
        print(f"epoch {epoch}")

        global_step, loss, acc = task.train(
            dl=train_dl,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            device=device,
            steps=global_step,
            epoch=epoch
        )
        losses.append(loss)
        train_accuracy.append(acc)
        train_epochs.append(epoch)

        print(f"loss:{loss}\taccuracy:{acc}")

        if epoch == 0 or epoch % args.eval_every == 0:
            ret = task.eval(
                test_dl,
                model,
                device
            )

            acc, prec, rec, f_score = ret.get("scores")
            print(f"--> eval - acc:{acc}\tprec:{prec}\trec:{rec}\tf1:{f_score}")

            eval_epochs.append(epoch)
            eval_accuracy.append(acc)
            eval_precision.append(prec)
            eval_recall.append(rec)
            eval_f_score.append(f_score)

            if f_score > best_f1:
                best_f1 = f_score
                state_dict = model.state_dict()
                state_dict = dict([(k, v.cpu()) for k, v in state_dict.items()])

                save_checkpoint(
                    path_=path.join(
                        settings.get("ckp_dir"),
                        exp_name
                    ),
                    state=state_dict,
                    is_best=False
                )

    train_timeseries = [
        create_timeseries(t, n, train_epochs) for t, n in [
            (train_accuracy, "accuracy"), (losses, "loss")
        ]
    ]
    plot_scalars(
        path.join(
            settings.get("run_dir"),
            exp_name,
            "train.png"
        ),
        train_timeseries
    )

    test_timeseries = [
        create_timeseries(t, n, eval_epochs) for t, n in [
            (eval_accuracy, "accuracy"),
            (eval_precision, "precision"),
            (eval_recall, "recall"),
            (eval_f_score, "f_score")
        ]
    ]
    plot_scalars(
        path.join(
            settings.get("run_dir"),
            exp_name,
            "test.png"
        ),
        test_timeseries
    )

    digit_one, _ = test[5]
    task.uncertanty_rotate(
        model,
        digit_one,
        path_=path.join(
            settings.get("run_dir"),
            exp_name,
            "rotate_img.png"
        ),
        device=device,
        threshold=.7
    )