from blockchain_ai.database.funder_ledger import FunderLedger


def test_funded_count_zero_for_unknown_funder(tmp_path):
    ledger = FunderLedger(str(tmp_path / "ledger.db"))
    assert ledger.funded_count("0xnobody", "0xwalleta") == 0


def test_record_then_funded_count_excludes_self(tmp_path):
    ledger = FunderLedger(str(tmp_path / "ledger.db"))
    ledger.record("0xfunder", "0xwalleta")
    assert ledger.funded_count("0xfunder", "0xwalleta") == 0


def test_funded_count_counts_other_distinct_wallets(tmp_path):
    ledger = FunderLedger(str(tmp_path / "ledger.db"))
    ledger.record("0xfunder", "0xwalleta")
    ledger.record("0xfunder", "0xwalletb")
    ledger.record("0xfunder", "0xwalletc")
    assert ledger.funded_count("0xfunder", "0xwalleta") == 2


def test_record_is_idempotent(tmp_path):
    ledger = FunderLedger(str(tmp_path / "ledger.db"))
    ledger.record("0xfunder", "0xwalleta")
    ledger.record("0xfunder", "0xwalleta")
    ledger.record("0xfunder", "0xwalletb")
    assert ledger.funded_count("0xfunder", "0xwalletb") == 1


def test_addresses_are_normalized_to_lowercase(tmp_path):
    ledger = FunderLedger(str(tmp_path / "ledger.db"))
    ledger.record("0xFunder", "0xWalletA")
    assert ledger.funded_count("0XFUNDER", "0xother") == 1


def test_ledger_persists_across_instances(tmp_path):
    db_path = str(tmp_path / "ledger.db")
    FunderLedger(db_path).record("0xfunder", "0xwalleta")
    ledger2 = FunderLedger(db_path)
    assert ledger2.funded_count("0xfunder", "0xwalletb") == 1
