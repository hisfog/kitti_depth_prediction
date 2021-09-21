# Sulotion:
## data_loader:
path image -> PIL.Image.open() ->
PIL.Image.Image -> torchvision.transforms.Compose([transforms.resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]) ->
torch.tensor with shape=[3,224,224] -> img.unsqueeze(0) ->
torch.tensor with shape=[1,3,224,224] -> model ->
...
## data_augment:
### random_crop:
ndarray -> ndarray
### rotate_image:
PIL.Image -> PIL.Image
### random_flip:
ndarray -> ndarray
### gamma, brightness, color augmention:
ndarray -> ndarray



