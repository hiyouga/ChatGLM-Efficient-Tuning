from huggingface_hub import snapshot_download
import os
repos_id = 'THUDM/chatglm-6b-int4' # original model 'THUDM/chatglm-6b' quantized 4bit 'THUDM/chatglm-6b-int4' 8bit 'THUDM/chatglm-6b-int8'
download_dir='./'+repos_id
snapshot_download(repo_id=repos_id, local_dir=download_dir, repo_type='model')