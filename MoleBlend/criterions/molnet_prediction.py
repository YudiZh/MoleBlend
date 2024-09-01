from dataclasses import dataclass, field

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from fairseq import metrics

@dataclass
class MolPredictionConfig(FairseqDataclass):
    tpu: bool = II("common.tpu")
    
    readout_type: str = field(
        default="sum",
        metadata={
            "help": "the readout type of the molnet finetune",
        },
    )


def get_roc_auc_score(meters):
    num_tasks = meters["num_tasks"].val
    predictions = meters["_predictions"].nparray
    labels = meters["_labels"].nparray
    weights = (meters["_weights"].nparray).reshape(-1, num_tasks)
    if meters["num_tasks"].val > 1:
        valid_preds = [[] for _ in range(num_tasks)]
        valid_targets = [[] for _ in range(num_tasks)]
        for i in range(labels.shape[0]):
            for j in range(num_tasks):
                if weights[i][j] != 0:
                    valid_preds[j].append(predictions[i][j])
                    valid_targets[j].append(labels[i][j])
        
        results = []
        for i in range(num_tasks):
            nan = False
            if all(target == 0 for target in valid_targets[i]) or all(target == 1 for target in valid_targets[i]):
                nan = True
            if len(valid_targets[i]) == 0:
                nan = True
            if nan:
                results.append(float('nan'))
                continue
            y_pred = 1/(1 + np.exp(-np.array(valid_preds[i])))
            results.append(roc_auc_score(y_true=valid_targets[i], y_score=y_pred))
        ras = np.nanmean(results)
    else:
        nan = False
        if all(labels==0) or all(labels==1):
            nan = True
        if len(labels) == 0:
            nan = True
        if nan:
            ras = float('nan')
        else:
            y_pred = 1/(1 + np.exp(-np.array(predictions))) # [B, ]
            ras = roc_auc_score(y_true=labels, y_score=predictions)
    return ras

def get_rmse_score(meters):
    predictions = meters["_predictions"].nparray
    labels = meters["_labels"].nparray
    rmse = mean_squared_error(y_true=labels, y_pred=predictions, squared=False)
    return rmse

def get_mae_score(meters):
    predictions = meters["_predictions"].nparray
    labels = meters["_labels"].nparray
    mae = mean_absolute_error(y_true=labels, y_pred=predictions)
    return mae

@register_criterion("mol_prediction", dataclass=MolPredictionConfig)
class MolPredictionLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked graph model (MGM) training.
    """

    def __init__(self, cfg: MolPredictionConfig, task):
        super().__init__(task)  
        self.moltask_type = task.moltask_type
        if self.moltask_type == 'classification':
            self.criterion = nn.BCEWithLogitsLoss(reduction = "none")
            # self.num_task = task.num_tasks
        else:
            self.criterion = nn.MSELoss()
            self.std = task.mol.dataset_train.dataset.std
            self.mean = task.mol.dataset_train.dataset.mean

        self.num_tasks = task.num_tasks
        self.readout_type = cfg.readout_type

    def half(self):
        self.fp16=True
        return super().half()

    def forward(self, model, sample):
        sample_size = 1
        with torch.no_grad():
            natoms = sample["net_input"]["batched_data"]['x'].shape[1]

        model_output = model(**sample["net_input"])

        logits = model_output[0]  
        

        logits_padding = sample['net_input']['batched_data']['x'][:, :, 0].eq(0).unsqueeze(-1)

        if self.readout_type == 'cls':
            logits = logits[:, 0, :]
        elif self.readout_type == 'mean':
            logits = logits[:, 1:, :].masked_fill_(logits_padding, 0.0).mean(dim=1)
        else:
            logits = logits[:, 1:, :].masked_fill_(logits_padding, 0.0).sum(dim=1)
        targets = model.get_targets(sample, [logits])

        weights = sample['net_input']['batched_data']['weights'].reshape(-1, self.num_tasks)

        if self.moltask_type == 'classification':
            if self.num_tasks == 1:	
                logits = logits.squeeze(-1)
                targets = targets.squeeze(-1)
            loss = self.criterion(logits, targets)
            mask = (weights != 0)
            if self.num_tasks == 1:	
                mask = mask.squeeze(-1)
            loss = loss * mask
            loss = loss.sum() / mask.sum()
        else: # reg
            new_logits = logits * self.std + self.mean
            loss = self.criterion(new_logits.view(-1), targets.view(-1))
            
        

        logging_output = {
            "loss": loss.item(),
            "sample_size": sample_size,
            "nsentences": sample_size,
            "ntokens": natoms,
        }
        num_tasks = self.num_tasks
        labels = targets.cpu().numpy()
        predictions = logits.cpu().detach().numpy()

        if not model.training:
            predictions = logits.cpu().detach().numpy()
            if self.moltask_type == 'regression':
                predictions = predictions * self.std + self.mean
            logging_output = {
                "loss": loss.item(),
                "sample_size": sample_size,
                "predictions": predictions,
                "num_tasks": num_tasks,
                "labels": labels,
                "weights": sample['net_input']['batched_data']['weights'].cpu().numpy(),
                "task_type": self.moltask_type,
                "nsentences": sample_size,
                "ntokens": natoms,
            }

        return loss, sample_size, logging_output


    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        total_loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        

        metrics.log_scalar(
            "loss", total_loss_sum / sample_size, sample_size, round=6
        )
        
        
        if "predictions" in logging_outputs[0]:
            num_tasks = logging_outputs[0]["num_tasks"]
            predictions = np.concatenate([log.get("predictions", np.array([]))for log in logging_outputs], axis=0)
            labels = np.concatenate([log.get("labels", np.array([]))for log in logging_outputs], axis=0)
            weights = np.concatenate([log.get("weights", np.array([]))for log in logging_outputs], axis=0)

            task_type = logging_outputs[0]["task_type"]
            if task_type == 'classification':
                metrics.log_scalar("num_tasks", num_tasks, 1, round=0)
                metrics.log_np_array("_predictions", predictions)
                metrics.log_np_array("_labels", labels)
                metrics.log_np_array("_weights", weights)
                metrics.log_derived(
                    "roc_auc",
                    get_roc_auc_score,
                )
            else:
                metrics.log_scalar("num_tasks", num_tasks, 1, round=0)
                metrics.log_np_array("_predictions", predictions)
                metrics.log_np_array("_labels", labels)
                metrics.log_np_array("_weights", weights)
                metrics.log_derived(
                    "rmse",
                    get_rmse_score,
                )
                metrics.log_derived(
                    "mae",
                    get_mae_score,
                )



    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False