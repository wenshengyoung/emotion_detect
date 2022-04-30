import torch
from repvgg import repvgg_model_convert, create_RepVGG_A0
train_model = create_RepVGG_A0(deploy=False)
train_model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load('./utils/train_model_30.pth').items()})
deploy_model = repvgg_model_convert(train_model, save_path='./utils/deploy_model_30.pth')

