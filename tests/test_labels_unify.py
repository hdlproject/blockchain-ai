from pathlib import Path
import csv
from blockchain_ai.connector.schema import AddressRecord, write_address_csv
from blockchain_ai.connector.unify import unify_addresses


def _addr(address, label, confidence, sources, flags):
    return AddressRecord(address, 1, label, confidence, sources, flags, "2026-01-01T00:00:00+00:00")


def test_deduplicates_same_address(tmp_path):
    f1 = tmp_path / "a.csv"
    f2 = tmp_path / "b.csv"
    write_address_csv([_addr("0xaaa", "sanctioned", 1.0, ["ofac"], ["ofac_sdn"])], f1)
    write_address_csv([_addr("0xaaa", "scammer", 0.8, ["goplus"], ["phishing"]),
                       _addr("0xbbb", "unknown", 0.0, ["goplus"], [])], f2)
    out = tmp_path / "unified.csv"
    records = unify_addresses([f1, f2], out)
    assert len(records) == 2


def test_merges_flags_and_sources(tmp_path):
    f1, f2 = tmp_path / "a.csv", tmp_path / "b.csv"
    write_address_csv([_addr("0xccc", "sanctioned", 1.0, ["ofac"], ["ofac_sdn"])], f1)
    write_address_csv([_addr("0xccc", "scammer", 0.9, ["goplus"], ["phishing_contract"])], f2)
    out = tmp_path / "out.csv"
    records = unify_addresses([f1, f2], out)
    r = records[0]
    assert set(r.sources) == {"ofac", "goplus"}
    assert "ofac_sdn" in r.flags
    assert "phishing_contract" in r.flags


def test_keeps_highest_confidence_label(tmp_path):
    f1, f2 = tmp_path / "a.csv", tmp_path / "b.csv"
    write_address_csv([_addr("0xddd", "suspicious", 0.4, ["goplus"], ["low"])], f1)
    write_address_csv([_addr("0xddd", "sanctioned", 1.0, ["ofac"], ["ofac_sdn"])], f2)
    out = tmp_path / "out.csv"
    records = unify_addresses([f1, f2], out)
    assert records[0].label == "sanctioned"
    assert records[0].confidence == 1.0


def test_writes_csv(tmp_path):
    f = tmp_path / "a.csv"
    write_address_csv([_addr("0xeee", "scammer", 0.8, ["goplus"], ["cybercrime"])], f)
    out = tmp_path / "out.csv"
    unify_addresses([f], out)
    assert out.exists()
    with open(out) as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["address"] == "0xeee"


def test_skips_missing_file(tmp_path):
    out = tmp_path / "out.csv"
    records = unify_addresses([tmp_path / "missing.csv"], out)
    assert records == []
