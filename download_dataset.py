import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download

# 数据集仓库ID
repo_id = "trixyL/simplestories-4k-megatron"

# 下载整个数据集到当前目录下的 simplestories-4k-megatron 文件夹
local_dir = "./simplestories-4k-megatron"

print(f"正在从 {repo_id} 下载数据集到 {local_dir} ...")
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",  # 重要：指定下载的是数据集，而非模型
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # 直接保存文件，而非创建符号链接
)
print("下载完成！")