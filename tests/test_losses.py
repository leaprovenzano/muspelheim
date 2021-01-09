import pytest
from hearth.losses import BinaryFocalLoss, MulticlassFocalLoss


@pytest.mark.parametrize('loss_type,', [BinaryFocalLoss, MulticlassFocalLoss])
def test_init_with_bad_reduction_fails(loss_type):
    expected_msg = (
        f'reduction \'sally\' is not supported for {loss_type.__name__},'
        ' please choose one of \\[\'mean\', \'sum\', \'none\'\\]'
    )
    with pytest.raises(ValueError, match=expected_msg):
        loss_type(reduction='sally')
