import torch
from evaluate import select_mnist_model, select_cifar_model
model_type = "small"
eps = "8px"
data = "cifar"
l = "inf"

if data == "mnist":
    select_model = select_mnist_model
elif data == "cifar":
    select_model = select_cifar_model

models = []
dir="./../models/seq_trained/l_"+str(l)+"/"
name_seq = str(data)+"_"+str(model_type)+"_"+str(eps)+".pth"
# name_seq = str(data)+"_"+str(model_type)+".pth"
d = torch.load(dir+name_seq)
sd = d['state_dict'][0]
m = select_model(model_type)
m.load_state_dict(sd)
models.append(m)

for id in range(2):
    dir="./../models/non_seq_trained/l_"+str(l)+"/"
    name = "more_"+str(data)+"_"+str(model_type)+"_"+str(eps)+"_"+str(id+1)+".pth"
    d = torch.load(dir+name)
    sd = d['state_dict'][0]
    m = select_model(model_type)
    m.load_state_dict(sd)
    models.append(m)
print("number of models: ", len(models))
torch.save({
    'state_dict' : [m.state_dict() for m in models], 
    'epoch' : 60,
    }, "./../models/non_seq_trained/"+name_seq)
