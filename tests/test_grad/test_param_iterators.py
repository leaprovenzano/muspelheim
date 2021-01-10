from torch import nn
from hearth.grad import named_trainable_parameters, freeze


def test_named_trainable_parameters():
    model = nn.Sequential(nn.Linear(10, 20), nn.Linear(10, 20), nn.Linear(20, 3))
    freeze(model[0])

    expect_names = ['1.weight', '1.bias', '2.weight', '2.bias']

    for expected_name, (name, param) in zip(expect_names, named_trainable_parameters(model)):
        assert name == expected_name
        assert param.requires_grad
        i, typ = name.split('.')
        assert (param == getattr(model[int(i)], typ)).all()
