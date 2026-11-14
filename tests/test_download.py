"""The source list of the download script."""
import download_sources as D


def test_source_list():
    assert len(D.FILES) == 6
    for rel, url, digest, size in D.FILES:
        assert url.startswith("https://") and len(digest) == 64 and size > 0
        assert "/main/" not in url and "/master/" not in url        # every URL is pinned to a revision


def test_check_reports_missing_files(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(D, "RAW", tmp_path)
    try:
        D.main(["--check"])
    except SystemExit as e:
        assert "6 file(s)" in str(e)
    else:
        raise AssertionError("missing files were not reported")
