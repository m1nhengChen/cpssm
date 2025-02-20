from .transformer import GraphTransformer
from omegaconf import DictConfig
from .brainnetcnn import BrainNetCNN
from .fbnetgen import FBNETGEN
from .gcn import MSGCN
from .BNT import BrainNetworkTransformer
from .COMTF import ComBrainTF
from .GBT import GeometricBrainTransformer
from .CPSSM import CorePeripherySSM
def model_factory(config: DictConfig):
    if config.model.name in ["LogisticRegression", "SVC"]:
        return None
    return eval(config.model.name)(config).cuda()
