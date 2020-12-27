import os
from hearth._file_utils import mkdirs_if_not_exist


def test_mkdirs_if_not_exist(tmpdir):
    parent = str(tmpdir.mkdir('models'))
    path = os.path.join(parent, 'boop', 'bop')
    mkdirs_if_not_exist(path, verbose=1)
    assert os.path.exists(path)
    assert os.path.isdir(path)
