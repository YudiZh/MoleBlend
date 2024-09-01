export NCCL_ASYNC_ERROR_HADNLING=1
export OMP_NUM_THREADS=1

python -m torch.distributed.launch --nproc_per_node=4 --master_port=10051 train.py \
        --user-dir ./MoleBlend \
        --data-path $data \
        --num-workers 8 --ddp-backend=legacy_ddp \
        --dataset-name PCQM4M-LSC-V2-3D \
        --batch-size 1024 --data-buffer-size 20 \
        --task graph_prediction --criterion graph_prediction --arch moleblend_base --num-classes 1 \
        --lr 1e-5 --warmup-init-lr 0 --min-lr 0 --lr-scheduler cosine \
        --warmup-updates 20000 --max-update 200000 --update-freq 1 \
        --encoder-layers 12 --encoder-attention-heads 32 --add-3d --num-3d-bias-kernel 128 \
        --encoder-embed-dim 768 --encoder-ffn-embed-dim 768 --droppath-prob 0.1 \
        --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 --weight-decay 0.0 \
        --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5 \
        --log-interval 50 --wandb-project $wandb_project --tensorboard-logdir ./$log_dir/$save_prefix/tsb \
        --save-dir ./$log_dir/$save_prefix --noise-scale 0.2 \
        --blending --blend-pred-spd --blend-pred-3d --blend-pred-edge --blend-prob '0.2,0.2,0.6' \
        --pred-3d-loss-factor $blend_3d_loss_factor --pred-spd-loss-factor $blend_spd_loss_factor --pred-edge-loss-factor $blend_edge_loss_factor \
        --fp16 \
        --regularization-3d-denosing \
        --denoising-3d-loss-factor $reg_3d_denoise \
        --disable-validation \
        --keep-last-epochs 50
