from softgriplab.provenance import text_sha256


def test_text_provenance_survives_checkout_but_detects_changed_data(tmp_path):
    a, b, changed = [
        tmp_path / name for name in ("unix.csv", "windows.csv", "changed.csv")
    ]
    a.write_bytes(b"time_s,force_n\n0,1.25\n")
    b.write_bytes(b"time_s,force_n\r\n0,1.25\r\n")
    changed.write_bytes(b"time_s,force_n\n0,1.26\n")
    assert text_sha256(a) == text_sha256(b)
    assert text_sha256(a) != text_sha256(changed)
