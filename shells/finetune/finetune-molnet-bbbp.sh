[ -z "${data}" ] && data="./data/molecular_property_prediction"
[ -z "${pretrain_ckpt_file}" ] && pretrain_ckpt_file="pretrained.pt"

export save_prefix="finetune_molnet"

export dataset_arg=bbbp
export WANDB_PROJECT=moleblend_molnet
export WANDB_NAME=molnet/$dataset_arg/$save_prefix
export WANDB_LOGDIR=./logs/$dataset_arg/$save_prefix/wandb
mkdir -p $WANDB_LOGDIR
export CUDA_VISIBLE_DEVICES="0"

ulimit -c unlimited

export NCCL_ASYNC_ERROR_HADNLING=1
export OMP_NUM_THREADS=1

python -m torch.distributed.launch --nproc_per_node=1 --master_port=10088 train.py \
	--user-dir ./MoleBlend \
	--data-path $data \
	--num-workers 16 --ddp-backend=legacy_ddp \
	--dataset-name moleculenet:$dataset_arg \
	--valid-subset valid,test \
	--batch-size 128 --data-buffer-size 20 \
	--task graph_prediction --criterion mol_prediction --arch moleblend_base --num-classes 1 \
	--warmup-init-lr 0 --lr 1e-5 --min-lr 1e-9 --lr-scheduler cosine \
	--warmup-epoch 8 --max-epoch 80 --update-freq 1 \
	--encoder-layers 12 --encoder-attention-heads 32 --num-3d-bias-kernel 128 \
	--encoder-embed-dim 768 --encoder-ffn-embed-dim 768 --droppath-prob 0.0 \
	--attention-dropout 0.1 --act-dropout 0.0 --dropout 0.0 --weight-decay 5e-5 \
	--optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5 \
	--log-interval 50 \
	--tensorboard-logdir ./logs/$dataset_arg/$save_prefix/tsb --save-dir ./logs/$dataset_arg/$save_prefix \
	--no-save \
    --finetune-from-model $pretrain_ckpt_file \
	--readout-type cls --remove-head