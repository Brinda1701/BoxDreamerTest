import os
import torch
from torchvision import models
from .base import PretrainedModelWrapper
_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]
class DinoV2Wrapper(PretrainedModelWrapper):
    def __init__(self, ckpt_path: None, cfg):
        super().__init__(model_name_or_path='dinov2')
        self.ckpt_path = ckpt_path
        # todo: support local path loading
        self.model_type = cfg.get('model_type', 'dinov2_vits14_reg')
        assert self.model_type in ['dinov2_vits14_reg', 'dinov2_vitb14_reg', 'dinov2_vitl14_reg', 'dinov2_vitg14_reg']
        
        self.freeze = cfg.get('freeze', True)
        self.device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model(self.device)
    
    def get_device(self):
        return self.device
    
    def to_device(self, device):
        self.model = self.model.to(device)
        self.device = device
    
    def load_model(self, device=None):
        if device is None:
            device = self.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        # load backbone model
        
        # 尝试加载 DINOv2 预训练模型
        # 说明：如果在不能连外网的服务器上，建议提前将本地 ~/.cache/torch/hub/ 目录上传
        try:
            if self.ckpt_path is not None:
                print(f"--> [DINOv2] 正在从本地指定路径加载: {self.ckpt_path}")
                model = torch.hub.load(self.ckpt_path, self.model_type, trust_repo=True, source='local').to(device)
            else:
                print(f"--> [DINOv2] 正在加载预训练模型: {self.model_type}")
                # 默认尝试从 Torch Hub 在线/本地缓存加载
                model = torch.hub.load('facebookresearch/dinov2', self.model_type).to(device)
        except Exception as e:
            # 如果在线连接 GitHub 失败，尝试检查本地是否存在缓存的 repo
            hub_dir = torch.hub.get_dir()
            local_repo_dir = os.path.join(hub_dir, "facebookresearch_dinov2_main")
            if os.path.exists(local_repo_dir):
                print(f"--> [DINOv2 Warning] 在线加载失败({e})，检测到本地离线缓存目录: {local_repo_dir}，正在切换为离线加载...")
                model = torch.hub.load(local_repo_dir, self.model_type, source='local', trust_repo=True).to(device)
            else:
                print("\n" + "="*70)
                print("[DINOv2 加载失败诊断指引]:")
                print("云端服务器访问 GitHub / Meta 权重源网络受限。请按以下方法任选其一解决：")
                print("1. 将本地电脑的 C:\\Users\\brinda\\.cache\\torch\\hub\\ 整个目录压缩上传到服务器的 ~/.cache/torch/hub/")
                print("2. 或者在实例化 BoxDreamerModel 前传入 ckpt_path 指向本地 dinov2 代码库目录。")
                print("="*70 + "\n")
                raise e
            
        self.model = model
        
        if self.freeze:
            self.model.eval()
            # freeze the model
            for param in self.model.parameters():
                param.requires_grad = False
                
    def _resnet_normalize_image(self, img: torch.Tensor) -> torch.Tensor:
        return (img - torch.tensor(_RESNET_MEAN, device=img.device, requires_grad=False).view(1, 3, 1, 1)) / torch.tensor(_RESNET_STD, device=img.device, requires_grad=False).view(1, 3, 1, 1)
     
    def predict(self, input_tensor):
        flag = False
        if input_tensor.dim() == 5:
            # BTCHW
            B, T, C, H, W = input_tensor.size()
            input_tensor = input_tensor.flatten(0, 1)
            flag = True
        with torch.no_grad():
            input_tensor = self._resnet_normalize_image(input_tensor)
            ret = self.model.forward_features(input_tensor)['x_norm_patchtokens']
            if flag:
                ret = ret.view(B, T, *ret.shape[1:])
            
            return ret