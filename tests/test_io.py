"""Self-contained IO adapter tests (no external data files)."""
import io
import tarfile

import pandas as pd

from recoverr import io as rio


def _make_slam_tar(tmp_path, track="en_es", stamp="20190204"):
    # Minimal SLAM train file: header lines + token lines ending in 0/1.
    lines = []
    for u in range(3):
        for e in range(4):
            lines.append(f"# user:u{u}  days:{e*3.0}  format:reverse_translate  session:lesson")
            lines.append("tok1 Hello NOUN _ _ _ _ 0")
            lines.append(f"tok2 world NOUN _ _ _ _ {e % 2}")
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    tar_path = tmp_path / f"data_{track}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        info = tarfile.TarInfo(name=f"{track}.slam.{stamp}.train")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    return tar_path


def test_from_slam(tmp_path):
    tar_path = _make_slam_tar(tmp_path)
    tele = rio.from_slam(tar_path, track="en_es", max_users=2)
    f = tele.frame
    assert set(["unit", "seq", "outcome"]).issubset(f.columns)
    for c in ["ex_id", "days_bin", "format", "token_freq"]:
        assert c in f.columns
    assert tele.n_units == 2                 # max_users respected
    assert f["outcome"].isin([0, 1]).all()   # binary error
    assert (f.groupby("unit")["seq"].apply(lambda s: s.is_monotonic_increasing)).all()


def test_from_hlr(tmp_path):
    df = pd.DataFrame({
        "user": ["a", "a", "b", "b"],
        "lexeme": ["x", "x", "y", "y"],
        "ts": [1, 2, 1, 2],
        "seen": [4, 4, 5, 5],
        "correct": [3, 2, 5, 4],
        "dt_days": [1.0, 2.0, 1.0, 3.0],
    })
    p = tmp_path / "sessions.parquet"
    df.to_parquet(p)
    tele = rio.from_hlr(p)
    assert set(["unit", "seq", "outcome"]).issubset(tele.frame.columns)
    assert tele.frame["outcome"].between(0, 1).all()
    tele_b = rio.from_hlr(p, binarize=True)
    assert tele_b.frame["outcome"].isin([0, 1]).all()


def test_from_chess(tmp_path):
    df = pd.DataFrame({
        "player": ["p1", "p1", "p1", "p2", "p2"],
        "ply": [1, 2, 3, 1, 2],
        "blunder": [0, 1, 0, 1, 0],
        "elo": [1500, 1500, 1500, 1800, 1800],
    })
    for ext, writer in [("parquet", df.to_parquet), ("csv", df.to_csv)]:
        fp = tmp_path / f"chess.{ext}"
        writer(fp, index=False) if ext == "csv" else writer(fp)
        tele = rio.from_chess(fp, covariates=["elo"])
        assert set(["unit", "seq", "outcome", "elo"]).issubset(tele.frame.columns)
        assert tele.n_units == 2
        assert tele.frame["outcome"].isin([0, 1]).all()
