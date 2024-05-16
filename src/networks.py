"""
The code is adapted from the following repository:
https://github.com/clinicalml/human_ai_deferral
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
from torchvision.models import DenseNet121_Weights, ResNet50_Weights


class WideResNet(nn.Module):
    """
    complex CNN model, gets 90% accuracy on CIFAR-10 without data-aug and 96% with
    here is how to create it: WideResNet(28, n_dataset + 1, 4, dropRate=0, hiden_dim) where hidden_dim is the dimension of the last layer
    the repr function extracts the last layer representation
    """

    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, hidden_dim=50):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(
            3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        self.nChannels = nChannels[3]
        self.softmax = nn.Softmax()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        out = self.fc(out)
        out = self.fc2(out)
        return out

    def repr(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        out = self.fc(out)
        return out


class DenseNet121_CE(nn.Module):
    """Model for just classification.
    The architecture of our model is the same as standard DenseNet121
    """

    def __init__(self, out_size):
        super(DenseNet121_CE, self).__init__()
        self.densenet121 = torchvision.models.densenet121(
            weights=DenseNet121_Weights.DEFAULT
        )
        self.num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(nn.Linear(self.num_ftrs, out_size))

    # def repr(self, x):
    # get representation before the last layer
    #    x = self.densenet121.features(x)
    #    return x

    def forward(self, x):
        x = self.densenet121(x)
        return x


class ResNet50_CE(nn.Module):
    """Model for just classification.
    The architecture of our model is the same as standard DenseNet121
    """

    def __init__(self, out_size):
        super(ResNet50_CE, self).__init__()
        self.resnet50 = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(nn.Linear(self.num_ftrs, out_size))

    # def repr(self, x):
    # get representation before the last layer
    #    x = self.densenet121.features(x)
    #    return x

    def forward(self, x):
        x = self.resnet50(x)
        return x


class BasicBlock(nn.Module):
    """
    Block for WideResNet
    """

    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = (
            (not self.equalInOut)
            and nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
            or None
        )

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, dropRate
        )

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(
                block(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    dropRate,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


# simple conv network
# (argument 2 of the first nn.Conv2d, and argument 1 of the second nn.Conv2d – they need to be the same number)
class NetSimple(nn.Module):
    """
    Simple 2 layer CNN with fully connected relu layers
    NetSimple(n_dataset) instantiates one such model
    with paramters to the max, this can get close to 80% accuracy
    """

    def __init__(self, num_classes, width1=6, width2=16, ff_units1=120, ff_units2=84):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width1, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(width1, width2, 5)
        self.fc1 = nn.Linear(width2 * 5 * 5, ff_units1)
        self.fc2 = nn.Linear(ff_units1, ff_units2)
        self.fc3 = nn.Linear(ff_units2, num_classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def repr(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class NetSimpleDefer(nn.Module):
    """
    Super model for L_{CE} loss that combines two NetSimple classifiers
    """

    def __init__(self, params_h, params_r):
        super().__init__()
        self.net_h = NetSimple(*params_h)
        self.net_r = NetSimple(*params_r)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x_h = self.net_h(x)
        x_r = self.net_r(x)
        x = torch.cat((x_h, x_r), 1)
        return x


class CNNText(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        n_filters,
        filter_sizes,
        output_dim,
        dropout,
        pad_idx,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.conv_0 = nn.Conv2d(
            in_channels=1,
            out_channels=n_filters,
            kernel_size=(filter_sizes[0], embedding_dim),
        )
        self.conv_1 = nn.Conv2d(
            in_channels=1,
            out_channels=n_filters,
            kernel_size=(filter_sizes[1], embedding_dim),
        )
        self.conv_2 = nn.Conv2d(
            in_channels=1,
            out_channels=n_filters,
            kernel_size=(filter_sizes[2], embedding_dim),
        )
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved_0 = F.relu(self.conv_0(embedded).squeeze(3))
        conved_1 = F.relu(self.conv_1(embedded).squeeze(3))
        conved_2 = F.relu(self.conv_2(embedded).squeeze(3))
        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim=1))
        return self.fc(cat)


# https://github.com/huyvnphan/PyTorch_CIFAR10
__all__ = ["MobileNetV2", "mobilenet_v2"]


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend(
            [
                # dw
                ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        # CIFAR10
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],  # Stride 2 -> 1 for CIFAR-10
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        # END

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))

        # CIFAR10: stride 2 -> 1
        features = [ConvBNReLU(3, input_channel, stride=1)]
        # END

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(input_channel, output_channel, stride, expand_ratio=t)
                )
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


def mobilenet_v2(pretrained=False, progress=True, device="cpu", **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(
            script_dir + "/state_dicts/mobilenet_v2.pt", map_location=device
        )
        model.load_state_dict(state_dict)
    return model


class Linear_net_sig(nn.Module):
    """
    Linear binary classifier
    """

    def __init__(self, input_dim, out_dim=1):
        super(Linear_net_sig, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x


class LinearNetDefer(nn.Module):
    """
    Linear Classifier with out+1 units and no softmax
    """

    def __init__(self, input_dim, out_dim):
        super(LinearNetDefer, self).__init__()
        # an affine operation: y = Wx + b
        self.fc = nn.Linear(input_dim, out_dim + 1)

    def forward(self, x):
        out = self.fc(x)
        return out


class LinearNet(nn.Module):
    """
    Linear Classifier with out units and no softmax
    """

    def __init__(self, input_dim, out_dim):
        super(LinearNet, self).__init__()
        # an affine operation: y = Wx + b
        self.fc = nn.Linear(input_dim, out_dim)

    def forward(self, x):
        out = self.fc(x)
        return out


class NonLinearNet(nn.Module):
    """
    NonLinear Classifier
    """

    def __init__(self, input_dim, out_dim):
        super(NonLinearNet, self).__init__()
        self.fc_all = nn.Sequential(
            nn.Linear(input_dim, 250),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(250, 100),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(100, out_dim),
        )
        # add dropout

    def forward(self, x):
        out = self.fc_all(x)

        return out


class ModelPredictAAE:
    def __init__(self, modelfile, vocabfile):
        """
        Edited from https://github.com/slanglab/twitteraae
        """
        self.vocabfile = vocabfile
        self.modelfile = modelfile
        self.load_model()

    def load_model(self):
        self.N_wk = np.loadtxt(self.modelfile)
        self.N_w = self.N_wk.sum(1)
        self.N_k = self.N_wk.sum(0)
        self.K = len(self.N_k)
        self.wordprobs = (self.N_wk + 1) / self.N_k
        self.vocab = [
            L.split("\t")[-1].strip() for L in open(self.vocabfile, encoding="utf8")
        ]
        self.w2num = {w: i for i, w in enumerate(self.vocab)}
        assert len(self.vocab) == self.N_wk.shape[0]

    def infer_cvb0(self, invocab_tokens, alpha, numpasses):
        doclen = len(invocab_tokens)

        # initialize with likelihoods
        Qs = np.zeros((doclen, self.K))
        for i in range(0, doclen):
            w = invocab_tokens[i]
            Qs[i, :] = self.wordprobs[self.w2num[w], :]
            Qs[i, :] /= Qs[i, :].sum()
        lik = Qs.copy()  # pertoken normalized but proportionally the same for inference

        Q_k = Qs.sum(0)
        for itr in range(1, numpasses):
            # print "cvb0 iter", itr
            for i in range(0, doclen):
                Q_k -= Qs[i, :]
                Qs[i, :] = lik[i, :] * (Q_k + alpha)
                Qs[i, :] /= Qs[i, :].sum()
                Q_k += Qs[i, :]

        Q_k /= Q_k.sum()
        return Q_k

    def predict_lang(self, tokens, alpha=1, numpasses=5, thresh1=1, thresh2=0.2):
        invocab_tokens = [w.lower() for w in tokens if w.lower() in self.w2num]
        # check that at least xx tokens are in vocabulary
        if len(invocab_tokens) < thresh1:
            return None
        # check that at least yy% of tokens are in vocabulary
        elif len(invocab_tokens) / len(tokens) < thresh2:
            return None
        else:
            posterior = self.infer_cvb0(
                invocab_tokens, alpha=alpha, numpasses=numpasses
            )
            # posterior is probability for African-American, Hispanic, Asian, and White (in that order)
            aae = (np.argmax(posterior) == 0) * 1
            return aae
