import csv
from blockchain_ai.connector.schema import (
    AddressRecord, TokenRecord,
    write_address_csv, write_token_csv,
    ADDRESS_FIELDNAMES, TOKEN_FIELDNAMES,
)


def test_address_record_to_row_pipes_sources_and_flags():
    r = AddressRecord(
        address="0xabc", chain_id=1, label="sanctioned", confidence=1.0,
        sources=["ofac", "goplus"], flags=["ofac_sdn", "scam_alert"],
        fetched_at="2026-01-01T00:00:00+00:00",
    )
    row = r.to_row()
    assert row["sources"] == "ofac|goplus"
    assert row["flags"] == "ofac_sdn|scam_alert"


def test_address_record_empty_lists_produce_empty_strings():
    r = AddressRecord("0xabc", 1, "unknown", 0.0, [], [], "2026-01-01T00:00:00+00:00")
    row = r.to_row()
    assert row["sources"] == ""
    assert row["flags"] == ""


def test_token_record_to_row():
    r = TokenRecord(
        token_address="0x123", chain_id=1, is_risky=True, risk_score=0.75,
        sources=["goplus"], flags=["honeypot"], fetched_at="2026-01-01T00:00:00+00:00",
    )
    row = r.to_row()
    assert row["token_address"] == "0x123"
    assert row["is_risky"] is True
    assert row["risk_score"] == 0.75


def test_write_address_csv_creates_file_with_header(tmp_path):
    records = [AddressRecord("0xaaa", 1, "sanctioned", 1.0, ["ofac"], ["ofac_sdn"], "2026-01-01T00:00:00+00:00")]
    out = tmp_path / "addr.csv"
    write_address_csv(records, out)
    assert out.exists()
    with open(out) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["address"] == "0xaaa"
    assert rows[0]["label"] == "sanctioned"


def test_write_token_csv_creates_file(tmp_path):
    records = [TokenRecord("0xbbb", 1, False, 0.0, ["goplus"], [], "2026-01-01T00:00:00+00:00")]
    out = tmp_path / "tok.csv"
    write_token_csv(records, out)
    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["is_risky"] == "False"
