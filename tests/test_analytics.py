import io

import pytest
from context import MSATools

EXPECTED = [('>seq1', 'AC-DEF'), ('>seq2', 'GH-IJK')]

# Same two wrapped entries with each line-ending style.
LINE_ENDINGS = {
    'lf': '>seq1\nAC-D\nEF\n>seq2\nGH-I\nJK\n',
    'crlf': '>seq1\r\nAC-D\r\nEF\r\n>seq2\r\nGH-I\r\nJK\r\n',
    'bare_cr': '>seq1\rAC-D\rEF\r>seq2\rGH-I\rJK\r',
}


@pytest.mark.parametrize('ending', LINE_ENDINGS)
def test_load_from_file_bytesio_line_endings(ending):
    msa = MSATools.load_from_file(io.BytesIO(LINE_ENDINGS[ending].encode()))
    assert msa.MSA == EXPECTED

@pytest.mark.parametrize('ending', LINE_ENDINGS)
def test_load_from_file_stringio_line_endings(ending):
    msa = MSATools.load_from_file(io.StringIO(LINE_ENDINGS[ending]))
    assert msa.MSA == EXPECTED

@pytest.mark.parametrize('ending', LINE_ENDINGS)
def test_load_from_file_path_line_endings(tmp_path, ending):
    # Write raw bytes so the file keeps the exact line endings under test.
    path = tmp_path / 'msa.afa'
    path.write_bytes(LINE_ENDINGS[ending].encode())
    assert MSATools.load_from_file(str(path)).MSA == EXPECTED
    assert MSATools.load_from_file(path).MSA == EXPECTED

def test_load_from_file_rejects_unsupported_source():
    with pytest.raises(TypeError):
        MSATools.load_from_file(12345)
