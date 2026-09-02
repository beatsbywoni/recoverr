"""Domain adapters: public datasets -> normalized Telemetry.

Each adapter reads a public source and returns a :class:`~recoverr.schema.Telemetry`
with unit/seq/outcome mapped as documented, so the same pipeline runs on all
validation domains. Ported from C1 (`c1_01_extract`) and C2 (`c2_01_build`).
"""
from __future__ import annotations

import io as _io
import os
import tarfile

import numpy as np
import pandas as pd

from .schema import Telemetry

SLAM_STAMP = "20190204"  # SLAM v4.0 train file stamp
SLAM_DAYS_BINS = (0.0, 7.0, 14.0, 30.0)


def from_slam(
    path,
    track: str = "en_es",
    stamp: str = SLAM_STAMP,
    max_users: int | None = None,
    days_bins=SLAM_DAYS_BINS,
) -> Telemetry:
    """Duolingo SLAM ``data_{track}.tar.gz`` -> token-level Telemetry.

    Mapping: unit=user, seq=pos_idx (per-user stream order), outcome=error
    (token marked incorrect). Covariates: ex_id, days_bin, format, token_freq.
    ``max_users`` streams only the first N learners (fast tests / smoke).
    """
    tok_rows, ex_meta = [], []
    ex_id = -1
    user = days = fmt = None
    seen: set[str] = set()
    skip = False
    with tarfile.open(str(path), "r:gz") as tf:
        # exact basename match avoids macOS AppleDouble "._" sidecar members
        member = next(m for m in tf.getmembers()
                      if os.path.basename(m.name) == f"{track}.slam.{stamp}.train")
        f = _io.TextIOWrapper(tf.extractfile(member), encoding="utf-8", errors="replace")
        for line in f:
            if line.startswith("# user:"):
                ex_id += 1
                meta = dict(p.split(":", 1) for p in line[2:].split() if ":" in p)
                user = meta["user"]
                if max_users is not None and user not in seen and len(seen) >= max_users:
                    skip = True
                    continue
                skip = False
                seen.add(user)
                days = float(meta.get("days", "nan"))
                fmt = meta.get("format", "")
                ex_meta.append((ex_id, user, days, fmt))
            elif line.startswith("#") or skip:
                continue
            else:
                cols = line.split()
                if len(cols) >= 3 and cols[-1] in ("0", "1"):
                    tok_rows.append((ex_id, user, cols[1].lower(), int(cols[-1])))

    tok = pd.DataFrame(tok_rows, columns=["ex_id", "user", "token", "error"])
    ex = pd.DataFrame(ex_meta, columns=["ex_id", "user", "days", "format"])
    tok["pos_idx"] = tok.groupby("user").cumcount()
    tok["token_freq"] = tok["token"].map(tok["token"].value_counts())
    tok = tok.merge(ex[["ex_id", "days", "format"]], on="ex_id", how="left")
    tok["days_bin"] = pd.cut(tok["days"], list(days_bins), labels=False, include_lowest=True)
    return Telemetry.from_frame(
        tok, unit="user", seq="pos_idx", outcome="error",
        covariates=["ex_id", "days_bin", "format", "token_freq"],
    )


HLR_USECOLS = ["timestamp", "user_id", "learning_language", "lexeme_id", "lexeme_string",
               "history_seen", "session_seen", "session_correct"]


def from_hlr(
    path,
    binarize: bool = True,
    max_users: int | None = 300,
    min_rows: int = 1000,
    nrows: int | None = None,
) -> Telemetry:
    """Duolingo HLR learning traces (Settles & Meeder 2016) -> review-level Telemetry.

    Accepts the public ``settles.acl16.learning_traces.13m.csv.gz`` (or an
    uncompressed copy). Each row is one (user, lexeme) practice record.

    Mapping: unit=user, seq=review order within user (timestamp, stable),
    outcome=``miss`` (1 if the item was answered wrongly at least once in the
    session; with ``binarize=False`` the session miss *rate*). Covariates:
    ``lexeme`` (sub-stream id for ``RecoveryPipeline(stream="lexeme")``),
    ``session`` (user x timestamp), ``hist_seen_bin`` (prior exposures
    1-3 / 4-9 / 10-29 / 30+), ``lex_diff_bin`` (quartile of the lexeme's
    corpus-wide miss rate), ``pos`` (part of speech from the lexeme tag) and
    ``lang`` (learning language).

    Users are kept if they have at least ``min_rows`` records; ``max_users``
    keeps the most active ones (deterministic). ``nrows`` limits the rows read
    (smoke tests). A legacy sessions parquet (columns user, ts, seen, correct,
    ...) is still accepted.
    """
    if str(path).endswith(".parquet"):
        df = pd.read_parquet(path).copy()
        acc = (df["correct"] / df["seen"]).clip(0, 1)
        df["miss"] = (1.0 - acc)
        if binarize:
            df["miss"] = (df["miss"] > 0).astype(int)
        cov = [c for c in ["lexeme", "dt_days", "pos", "seq_k"] if c in df.columns]
        return Telemetry.from_frame(df, unit="user", seq="ts", outcome="miss", covariates=cov)

    df = pd.read_csv(path, usecols=HLR_USECOLS, nrows=nrows,
                     dtype={"user_id": "category", "lexeme_id": "category",
                            "learning_language": "category", "lexeme_string": "category"})
    counts = df["user_id"].value_counts()
    keep = counts[counts >= min_rows]
    if max_users is not None:
        keep = keep.sort_values(ascending=False, kind="stable").iloc[:max_users]
    df = df[df["user_id"].isin(keep.index)].copy()
    # lexeme difficulty from the *full* file read (corpus-wide miss rate quartile)
    df["miss_rate"] = 1.0 - (df["session_correct"] / df["session_seen"]).clip(0, 1)
    df["miss"] = (df["session_correct"] < df["session_seen"]).astype(int)
    lex_rate = df.groupby("lexeme_id", observed=True)["miss"].mean()
    df["lex_diff_bin"] = pd.qcut(df["lexeme_id"].map(lex_rate).astype(float), 4, labels=False, duplicates="drop")
    df["hist_seen_bin"] = pd.cut(df["history_seen"], [0, 3, 9, 29, np.inf], labels=False)
    df["pos"] = df["lexeme_string"].astype(str).str.extract(r"<([a-z]+)", expand=False).fillna("unk")
    df = df.sort_values(["user_id", "timestamp"], kind="stable")
    df["seq"] = df.groupby("user_id", observed=True).cumcount()
    df["session"] = (df["user_id"].astype(str) + ":" + df["timestamp"].astype(str))
    out = df.rename(columns={"user_id": "user", "lexeme_id": "lexeme", "learning_language": "lang"})
    outcome = "miss" if binarize else "miss_rate"
    return Telemetry.from_frame(out, unit="user", seq="seq", outcome=outcome,
                                covariates=["lexeme", "session", "hist_seen_bin", "lex_diff_bin", "pos", "lang"])


def from_chess(
    path,
    unit: str = "player",
    seq: str = "ply",
    outcome: str = "blunder",
    covariates=(),
) -> Telemetry:
    """Move-level chess table -> Telemetry (performance domain).

    Reads a ``.parquet`` or ``.csv`` table with a player id, a within-player move
    order, and a binary failure indicator (e.g. blunder = eval drop beyond a
    threshold). Column names are configurable via ``unit`` / ``seq`` / ``outcome``.
    To build such a table from the public Lichess database use
    :func:`parse_lichess_pgn` followed by :func:`from_lichess_moves`.
    """
    p = str(path)
    df = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
    return Telemetry.from_frame(df, unit=unit, seq=seq, outcome=outcome,
                                covariates=list(covariates))


# ---------------------------------------------------------------- Lichess PGN
_EVAL_RE = None


def _win_prob(cp):
    """Lichess win probability (0..1) from centipawns (mate -> +-inf)."""
    return 1.0 / (1.0 + np.exp(-0.00368208 * np.clip(cp, -1000, 1000)))


def _iter_pgn_lines(path):
    """Yield lines of a .pgn or .pgn.zst file (zstd via the ``zstandard`` module
    or the ``zstd`` command-line tool, whichever is available)."""
    import io as _io2
    import subprocess
    p = str(path)
    if p.endswith(".zst"):
        try:
            import zstandard as zstd
            with open(p, "rb") as fh:
                with zstd.ZstdDecompressor().stream_reader(fh) as reader:
                    for line in _io2.TextIOWrapper(reader, encoding="utf-8", errors="replace"):
                        yield line
            return
        except ImportError:
            proc = subprocess.Popen(["zstd", "-dc", p], stdout=subprocess.PIPE)
            for line in _io2.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace"):
                yield line
            proc.wait()
            return
    with open(p, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            yield line


def parse_lichess_pgn(path, max_games: int | None = None, hash_players: bool = True,
                      skip_games: int = 0) -> pd.DataFrame:
    """Stream a Lichess database PGN(.zst) and return one row per *move* of the
    games that carry Stockfish ``[%eval]`` annotations.

    Columns: game_id, player (sha1-hashed username unless ``hash_players=False``),
    color, ply, move_no, elo, opp_elo, time_control, date, wp_before, wp_after
    (win probability for the mover, 0..1), wp_loss (= wp_before - wp_after,
    clipped at 0), nag (Lichess annotation: '' / '?!' / '?' / '??').
    Games are capped at ``max_games`` *annotated* games; ``skip_games`` skips the
    first N annotated games (chunked parsing of large monthly files; a full month
    holds ~10^7 moves, so parse in chunks and concatenate parquet parts).
    """
    import hashlib
    import re
    global _EVAL_RE
    if _EVAL_RE is None:
        _EVAL_RE = re.compile(r"(\d+)(\.{1,3})\s+(\S+?)([?!]{0,2})\s*\{\s*\[%eval\s+([^\]]+)\]")
    def _hash(name):
        return hashlib.sha1(name.encode("utf-8")).hexdigest()[:12] if hash_players else name
    rows, hdr, n_games = [], {}, 0
    for line in _iter_pgn_lines(path):
        if line.startswith("["):
            k, _, v = line[1:].partition(" ")
            hdr[k] = v.strip().rstrip("]").strip('"')
            continue
        if not line.startswith("1.") or "%eval" not in line:
            if line.strip() == "":
                pass
            elif line.startswith("1."):
                hdr = {}
            continue
        if n_games < skip_games:
            n_games += 1
            hdr = {}
            continue
        game_id = hdr.get("Site", "").rsplit("/", 1)[-1] or str(n_games)
        white, black = _hash(hdr.get("White", "?")), _hash(hdr.get("Black", "?"))
        try:
            welo, belo = int(hdr.get("WhiteElo", "0")), int(hdr.get("BlackElo", "0"))
        except ValueError:
            welo = belo = 0
        tc, date = hdr.get("TimeControl", "-"), hdr.get("UTCDate", "") + " " + hdr.get("UTCTime", "")
        prev_wp_white = 0.5  # win prob for White before the first move (eval 0)
        ply = 0
        for m in _EVAL_RE.finditer(line):
            ev = m.group(5).strip()
            if ev.startswith("#"):
                mate = int(ev[1:])
                cp = 100000.0 if mate > 0 else -100000.0
            else:
                try:
                    cp = float(ev) * 100.0
                except ValueError:
                    continue
            wp_white_after = float(_win_prob(cp))
            is_white = m.group(2) == "."
            wp_before = prev_wp_white if is_white else 1.0 - prev_wp_white
            wp_after = wp_white_after if is_white else 1.0 - wp_white_after
            ply += 1
            rows.append((game_id, white if is_white else black, "w" if is_white else "b", ply, int(m.group(1)),
                         welo if is_white else belo, belo if is_white else welo, tc, date,
                         wp_before, wp_after, max(0.0, wp_before - wp_after), m.group(4)))
            prev_wp_white = wp_white_after
        n_games += 1
        hdr = {}
        if max_games is not None and n_games - skip_games >= max_games:
            break
    df = pd.DataFrame(rows, columns=["game_id", "player", "color", "ply", "move_no", "elo", "opp_elo",
                                     "time_control", "date", "wp_before", "wp_after", "wp_loss", "nag"])
    for c in ("game_id", "player", "color", "time_control", "nag"):
        df[c] = df[c].astype("category")
    for c in ("ply", "move_no", "elo", "opp_elo"):
        df[c] = df[c].astype("int32")
    for c in ("wp_before", "wp_after", "wp_loss"):
        df[c] = df[c].astype("float32")
    return df


def from_lichess_moves(
    moves: pd.DataFrame,
    error_threshold: float = 0.10,
    min_moves: int = 500,
    max_players: int | None = None,
) -> Telemetry:
    """Move table from :func:`parse_lichess_pgn` -> per-player Telemetry.

    unit = player, seq = the player's own moves in chronological order (games by
    date, moves by ply), outcome = ``error`` (1 for a Lichess mistake ``?`` or
    blunder ``??``; if annotations are absent, a win-probability loss of at least
    ``error_threshold``). Covariates: game_id (episode / stream id), blunder
    (Lichess ``??``, or wp_loss >= .30 without annotations; the default failure anchor), phase
    (opening <= move 10, middlegame, endgame > move 30), wp_before_bin
    (quartile of the mover's win probability before the move, for matching on
    game state), wp_after_bin (same, after the move: a placebo matched on it
    starts from a position as good or bad as the one the blunder produced),
    elo_bin, time_control, color.
    """
    df = moves.copy()
    counts = df["player"].value_counts()
    keep = counts[counts >= min_moves]
    if max_players is not None:
        keep = keep.sort_values(ascending=False, kind="stable").iloc[:max_players]
    df = df[df["player"].isin(keep.index)].copy()
    has_nag = df["nag"].notna()
    # Lichess's own move classification when present ('?' mistake, '??' blunder);
    # win-probability thresholds otherwise
    df["error"] = np.where(has_nag, df["nag"].isin(["?", "??"]), df["wp_loss"] >= error_threshold).astype(int)
    df["blunder"] = np.where(has_nag, df["nag"] == "??", df["wp_loss"] >= 0.30).astype(int)
    df["phase"] = pd.cut(df["move_no"], [0, 10, 30, np.inf], labels=["opening", "middlegame", "endgame"]).astype(str)
    df["wp_before_bin"] = pd.cut(df["wp_before"], [-0.01, 0.25, 0.5, 0.75, 1.01], labels=False)
    df["wp_after_bin"] = pd.cut(df["wp_after"], [-0.01, 0.25, 0.5, 0.75, 1.01], labels=False)
    df["elo_bin"] = pd.cut(df["elo"], [0, 1200, 1500, 1800, 2100, 4000], labels=False)
    df = df.sort_values(["player", "date", "game_id", "ply"], kind="stable")
    df["seq"] = df.groupby("player", observed=True).cumcount()
    return Telemetry.from_frame(df, unit="player", seq="seq", outcome="error",
                                covariates=["game_id", "blunder", "phase", "wp_before_bin", "wp_after_bin", "elo_bin", "time_control", "color", "wp_loss"])
