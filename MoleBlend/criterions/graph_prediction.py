from dataclasses import dataclass, field
import math
from omegaconf import II

import torch
import torch.nn as nn
from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
import os
import torch.nn.functional as F


@dataclass
class GraphPredictionConfig(FairseqDataclass):
    tpu: bool = II("common.tpu")

    pred_spd_loss_factor: float = field(
        default=1.0,
        metadata={
            "help": "the loss factor of spd pred",
        }
    )

    pred_edge_loss_factor: float = field(
        default=1.0,
        metadata={
            "help": "the loss factor of edge pred",
        }
    )

    pred_3d_loss_factor: float = field(
        default=1.0,
        metadata={
            "help": "the loss factor of 3d pred"
        }
    )

    denoising_3d_loss_factor: float = field(
        default=1.0,
        metadata={
            "help": "the loss factor of 3d pred"
        }
    )




    sum_loss: bool = field(
        default=False,
    )
    
    

@register_criterion("graph_prediction", dataclass=GraphPredictionConfig)
class GraphPredictionLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked graph model (MGM) training.
    """

    def __init__(self, cfg: GraphPredictionConfig, task):
        super().__init__(task)
        self.tpu = cfg.tpu
        self.noise_scale = task.cfg.noise_scale
        self.pred_spd_loss_factor = cfg.pred_spd_loss_factor
        self.pred_edge_loss_factor = cfg.pred_edge_loss_factor
        self.pred_3d_loss_factor = cfg.pred_3d_loss_factor
        self.denoising_3d_loss_factor = cfg.denoising_3d_loss_factor
        self.fp16 = False
        self.sum_loss = cfg.sum_loss

    
    def half(self):
        self.fp16=True
        return super().half()

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = 1
        with torch.no_grad():
            natoms = sample["net_input"]["batched_data"]['x'].shape[1]

        # add gaussian noise
        if not self.task.cfg.regularization_3d_denosing:
            ori_pos = sample['net_input']['batched_data']['pos']
        else:
            ori_pos = sample['net_input']['batched_data']['pos']  # [B, n_atom, 3]
            noise = torch.randn(ori_pos.shape).to(ori_pos) * self.noise_scale
            noise_mask = (ori_pos == 0.0).all(dim=-1, keepdim=True)
            noise = noise.masked_fill_(noise_mask, 0.0)
            sample['net_input']['batched_data']['pos'] = ori_pos + noise

        model_output = model(**sample["net_input"])
        logits, node_output, blend_spd_pred_logtis, blend_edge_pred_logits, blend_3d_pred_logits = model_output[0], model_output[1], model_output[2], model_output[3], model_output[4]  # [B, n_atom+1, 1], [B, n_atom, 3]
        blend_mask_3d_feature = model_output[5]

        back_losses = []
        losses = []
        node_output_loss = None
        if self.task.cfg.regularization_3d_denosing:
            if node_output is not None:
                node_mask = (node_output == 0.0).all(dim=-1).all(dim=-1)[:, None, None] + noise_mask
                if blend_mask_3d_feature is not None:
                    node_mask = node_mask | (~blend_mask_3d_feature)
                node_output = node_output.masked_fill_(node_mask, 0.0)


                if self.fp16 is True:
                    node_output_loss = (1.0 - nn.CosineSimilarity(dim=-1)(node_output.to(torch.float32), noise.masked_fill_(node_mask, 0.0).to(torch.float32)))
                    node_output_loss = node_output_loss.masked_fill_(node_mask.squeeze(-1), 0.0).sum(dim=-1).to(torch.float16)
                else:
                    node_output_loss = (1.0 - nn.CosineSimilarity(dim=-1)(node_output.to(torch.float32), noise.masked_fill_(node_mask, 0.0).to(torch.float32)))
                    node_output_loss = node_output_loss.masked_fill_(node_mask.squeeze(-1), 0.0).sum(dim=-1).to(torch.float32)  # [B]

                tgt_count = (~node_mask).squeeze(-1).sum(dim=-1).to(node_output_loss) # [B,]
                
                if self.sum_loss:
                    tgt_count = tgt_count.masked_fill_(tgt_count == 0.0, 1.0)
                    node_output_loss = (node_output_loss / tgt_count).sum() * 1
                else:
                    node_output_loss = node_output_loss.sum()
                    tgt_count = tgt_count.sum()
                    node_output_loss = (node_output_loss / tgt_count)
            else:
                bsz = noise.size(0)
                if self.sum_loss:
                    node_output_loss = (noise - noise).sum()
                else:
                    node_output_loss = (noise - noise).mean()
            back_losses.append(self.denoising_3d_loss_factor * node_output_loss)
            losses.append(node_output_loss)
        
        spd_loss = None
        if self.task.cfg.blending and self.task.cfg.blend_pred_spd and blend_spd_pred_logtis is not None:
            merge_spd_dist_start = 11
            merge_spd_dist_end = 19
            spd_loss = self.compute_spd_blending_loss(sample, blend_spd_pred_logtis, merge_spd_dist_start, merge_spd_dist_end)
            back_losses.append(self.pred_spd_loss_factor * spd_loss) 
            losses.append(spd_loss)
        
        edge_loss = None
        if self.task.cfg.blending and self.task.cfg.blend_pred_edge and blend_edge_pred_logits is not None:
            edge_loss = self.compute_edge_blending_ce_loss(sample, blend_edge_pred_logits, model.args.edge_type_1_num_classes, model.args.edge_type_2_num_classes, model.args.edge_type_3_num_classes)
            back_losses.append(self.pred_edge_loss_factor * edge_loss)
            losses.append(edge_loss)

        loss_3d = None

        if self.task.cfg.blending and self.task.cfg.blend_pred_3d and blend_3d_pred_logits is not None:
            loss_3d = self.compute_3d_blending_loss(blend_3d_pred_logits, ori_pos)
            back_losses.append(self.pred_3d_loss_factor * loss_3d)
            losses.append(loss_3d)


        loss_backward = sum(back_losses)
        loss = sum(losses)
        
        logging_output = {
            "loss": loss.item(),
            "sample_size": sample_size,
            "nsentences": sample_size,
            "ntokens": natoms,
        }

        if node_output_loss is not None:
            logging_output["node_output_loss"] = node_output_loss.item()

        if spd_loss is not None:
            logging_output["spd_loss"] = spd_loss.item()
        
        if edge_loss is not None:
            logging_output["edge_loss"] = edge_loss.item()

        if loss_3d is not None:
            logging_output["3d_loss"] = loss_3d.item()

        return loss_backward, sample_size, logging_output

    def compute_spd_blending_loss(self, sample, blend_logits, merge_spd_dist_start, merge_spd_dist_end):
        target = sample["net_input"]["batched_data"]["spatial_pos"]  
        target = (target - 1).clamp(min=0) 
        merge_spd_dist_mask = (target >= merge_spd_dist_start) & (target <= merge_spd_dist_end)
        target = target.masked_fill_(merge_spd_dist_mask, merge_spd_dist_start)
        target -= 1

        disconnect_spd = (target == 509)|(target == -1)
        target = target.masked_fill_(disconnect_spd, -100)

        # generate spd mask tensor
        spd_loss = nn.CrossEntropyLoss(reduction='mean')(blend_logits.permute(0, 3, 1, 2).to(torch.float32), target) 
        if self.fp16:
            spd_loss = spd_loss.to(torch.float16)
        return spd_loss
    
    def compute_edge_blending_ce_loss(
        self, 
        sample, 
        blend_edge_logits, 
        num_classes_1,
        num_classes_2,
        num_classes_3,
    ):

        target_edge_input = sample['net_input']['batched_data']['edge_input'].clone()
        target_edge_input_dist_bool = (target_edge_input==0).all(dim=-1) 

        def revert_edge_feature_from_single_embed(x, offset: int = 512):
            feature_num = x.size(-1)
            feature_offset = (3 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)[None, None, None, None, :]).to(x.device)

            feature = (x - feature_offset).clamp(min=-1)
            return feature

        edge_feature = revert_edge_feature_from_single_embed(target_edge_input, offset=512)
        target_edge_feature = (num_classes_2 * num_classes_3)*edge_feature[:,:,:,:,0]+num_classes_3*edge_feature[:,:,:,:,1] + edge_feature[:,:,:,:,2]

        target_edge_feature = target_edge_feature.masked_fill_(target_edge_input_dist_bool, -100) # 
        assert ((target_edge_feature < 0) & (target_edge_feature!=-100)).sum()==0
        
        loss = nn.CrossEntropyLoss(reduction='mean')(blend_edge_logits.permute(0, 4, 1, 2, 3).to(torch.float32), target_edge_feature) 
        if self.fp16:
            loss = loss.to(torch.float16)
        
        return loss
    
    def compute_3d_blending_loss(
        self,
        blend_3d_logits, 
        ori_pos,
    ):
        target_3d = ori_pos.unsqueeze(1) - ori_pos.unsqueeze(2)
        orig_pos_mask = (ori_pos == 0.0).all(dim=-1, keepdim=True) 
        pred_3d_mask = (target_3d == 0.0).all(dim=-1) 
        pred_3d_mask = pred_3d_mask | orig_pos_mask | orig_pos_mask.transpose(-2,-1)

        blend_3d_logits = blend_3d_logits.masked_fill_(pred_3d_mask.unsqueeze(-1), 0.0)
        loss = (1.0 - nn.CosineSimilarity(dim=-1)(blend_3d_logits.to(torch.float32), target_3d.to(torch.float32)))

        if self.fp16:
            loss = loss.masked_fill_(pred_3d_mask, 0.0).sum(-1).sum(-1).to(torch.float16)
        else:
            loss = loss.masked_fill_(pred_3d_mask, 0.0).sum(-1).sum(-1).to(torch.float32)
        
        pred_3d_tgt_count = (~pred_3d_mask).sum(-1).sum(-1)
        tgt_3d_count_mask = (pred_3d_tgt_count==0)
        pred_3d_tgt_count = pred_3d_tgt_count.masked_fill_(tgt_3d_count_mask, 1.0)
        loss = loss / pred_3d_tgt_count
        loss = loss.masked_fill_(tgt_3d_count_mask, 0.0)
        pred_3d_tgt_sample_count = (~tgt_3d_count_mask).sum()

        if pred_3d_tgt_sample_count == 0 :
            pred_3d_tgt_sample_count = pred_3d_tgt_count.new_tensor(1)
        
        loss = loss.sum() / pred_3d_tgt_sample_count
        
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        total_loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        

        metrics.log_scalar(
            "loss", total_loss_sum / sample_size, sample_size, round=6
        )
        
        if "node_output_loss" in logging_outputs[0]:
            node_output_loss_sum = sum(log.get("node_output_loss", 0) for log in logging_outputs)
            metrics.log_scalar(
                "node_output_loss", node_output_loss_sum / sample_size, sample_size, round=6
            )

        if "spd_loss" in logging_outputs[0]:
            spd_loss_sum = sum(log.get("spd_loss", 0) for log in logging_outputs)
            metrics.log_scalar(
                "spd_loss", spd_loss_sum / sample_size, sample_size, round=6
            )
        
        if "edge_loss" in logging_outputs[0]:
            loss_edge_sum = sum(log.get("edge_loss") for log in logging_outputs)
            metrics.log_scalar(
                "edge_loss", loss_edge_sum / sample_size, sample_size, round=6
            )
        
        if "3d_loss" in logging_outputs[0]:
            loss_3d_sum = sum(log.get("3d_loss") for log in logging_outputs)
            metrics.log_scalar(
                "3d_loss", loss_3d_sum / sample_size, sample_size, round=6
            )
        

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
