# Single-node, NVLink only
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=PHB
nvidia-smi -pm 1