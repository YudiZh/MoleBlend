[ -z "${data}" ] && export data="./data/PCQM4Mv2"

export blend_3d_loss_factor=0.5
export blend_spd_loss_factor=0.1
export blend_edge_loss_factor=0.1
export reg_3d_denoise=6
export wandb_project="moleblend_pretrain"


export save_prefix="MoleBlend-lr_1e5-bsz4096_1024_4gpu-warm20k-3d${blend_3d_loss_factor}spd${blend_spd_loss_factor}edge${blend_edge_loss_factor}node${reg_3d_denoise}-blend226-fp16"

export WANDB_NAME=$save_prefix
export CUDA_VISIBLE_DEVICES='0,1,2,3'

export log_dir=logs
export WANDB_LOGDIR="./wandb"
mkdir -p $WANDB_LOGDIR

mkdir -p ./$log_dir/$save_prefix/
bash shells/pretrain-moleblend.sh
