import torch
import numpy as np
from argparse import Namespace
from torch.utils.data import DataLoader
from typing import List, Optional
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from src.model import BaseModel, edl_mse_loss, relu_evidence

class UncertaintyTask(object):
    def __init__(
            self,
            name: str,
    ):
        super(UncertaintyTask, self).__init__()
        self.name = name

    def train(
            self,
            dl: DataLoader,
            model: BaseModel,
            optimizer: torch.optim.Optimizer,
            scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
            device: torch.device,
            steps: int,
            epoch: int

    ):
        total_loss = 0
        total_correct = 0

        num_predictions = 0
        model.train()

        for idx, (x, labels_t) in enumerate(dl):
            x = x.to(device)
            labels_t = labels_t.to(device)
            optimizer.zero_grad()

            logits_t = model(x)
            loss_t, *_ = edl_mse_loss(
                logits_t,
                labels_t,
                epoch,
                model.conf.num_labels,
                annealing_step=10
            )

            loss_t = loss_t.mean()
            loss_t.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()
            steps += 1

            # update metrics
            _, preds_t = torch.max(logits_t, 1)
            match = torch.eq(preds_t, labels_t).float()
            total_correct += match.sum().item()
            num_predictions += labels_t.size(0)
            total_loss += loss_t.item()

        total_loss /= (idx + 1)
        accuracy = total_correct / num_predictions

        return steps, total_loss, accuracy

    def eval(
            self,
            dl: DataLoader,
            model: BaseModel,
            device: torch.device
    ):
        model.eval()

        labels = []
        preds = []

        for idx, (x, labels_t) in enumerate(dl):
            x = x.to(device)
            labels_t = labels_t.to(device)

            with torch.no_grad():
                logits_t = model(x)
                _, preds_t = torch.max(logits_t, 1)

                preds.append(preds_t.cpu().detach().numpy())
                labels.append(labels_t.cpu().detach().numpy())

        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(labels, axis=0)

        prec, rec, f_score, _ = precision_recall_fscore_support(
            labels,
            preds,
            average='macro'
        )

        acc = accuracy_score(
            labels,
            preds
        )

        return dict(
            scores=(acc, prec, rec, f_score)
        )

    def uncertanty_rotate(
            self,
            model: BaseModel,
            img: torch.Tensor,
            path_: str,
            device: torch.device,
            threshold: float = 0.5
    ) -> None:
        import torchvision.transforms as transforms
        import scipy.ndimage as nd
        import matplotlib.pyplot as plt

        model.eval()
        num_labels = model.conf.num_labels
        Mdeg = 180
        Ndeg = int(Mdeg / 10) + 1
        ldeg = []
        lp = []
        lu = []
        classifications = []

        def rotate_img(x, deg):
            return nd.rotate(x.reshape(28, 28), deg, reshape=False).ravel()

        scores = np.zeros((1, num_labels))
        rimgs = np.zeros((28, 28 * Ndeg))
        for i, deg in enumerate(np.linspace(0, Mdeg, Ndeg)):
            nimg = rotate_img(img.numpy()[0], deg).reshape(28, 28)

            nimg = np.clip(a=nimg, a_min=0, a_max=1)
            rimgs[:, i*28: (i + 1) * 28] = nimg
            trans = transforms.ToTensor()
            img_tensor = trans(nimg)
            img_tensor = img_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(img_tensor)

            evidence = relu_evidence(logits)
            alpha = evidence + 1
            uncertainty = num_labels / torch.sum(alpha, dim=1, keepdim=True)
            probs = alpha / torch.sum(alpha, dim=1, keepdim=True)

            _, preds = torch.max(logits, 1)
            probs = probs.flatten().detach().cpu()
            preds = preds.flatten().detach().cpu()
            classifications.append(preds[0].item())
            lu.append(uncertainty.mean().detach().cpu())

            scores += probs.numpy() > threshold
            ldeg.append(deg)
            lp.append(probs.tolist())

        labels = np.arange(10)[scores[0].astype(bool)]
        lp = np.array(lp)[:, labels]

        c = ["black", "blue", "red", "brown", "purple", "cyan"]
        marker = ["s", "^", "o"] * 2
        labels = labels.tolist()

        fig, axs = plt.subplots(
            nrows=3,
            gridspec_kw=dict(height_ratios=[4, 1, 12]),
            figsize=(6.2, 5)
        )

        for i in range(len(labels)):
            axs[2].plot(ldeg, lp[:, i], marker=marker[i], c=c[i])

        labels += ["uncertainty"]
        axs[2].plot(ldeg, lu, marker="<", c="red")

        classifications = [str(c) for c in classifications]
        print(" - ".join(classifications))

        axs[0].set_title("Rotated \"1\" Digit Classifications")
        axs[0].imshow(1-rimgs, cmap="gray")
        axs[0].axis("off")


        axs[1].table(
            cellText=[classifications],
            cellLoc="center",
            bbox=[0, 1.2, 1, 1]
        )
        axs[1].axis("off")

        axs[2].set_xlim([0, Mdeg])
        axs[2].set_ylim([0, 1])
        axs[2].set_xlabel("Rotation Degree")
        axs[2].set_ylabel("Classification Probability")
        axs[2].legend(labels)

        fig.savefig(path_)
