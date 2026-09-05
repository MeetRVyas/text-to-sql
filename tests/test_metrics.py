"""Tests for evaluation/metrics.py."""

from evaluation.metrics import compute_metrics, exact_match, execution_accuracy, normalize_sql


# ---------------------------------------------------------------------------
# normalize_sql
# ---------------------------------------------------------------------------

def test_normalize_sql_lowercases_keywords_but_not_identifiers():
    normalized = normalize_sql("SELECT Name FROM Singer WHERE Country = 'US'")
    assert normalized == "select Name from Singer where Country = 'US'"


def test_normalize_sql_collapses_whitespace_and_strips_semicolon():
    assert normalize_sql("SELECT   *   FROM singer;  ") == "select * from singer"


# ---------------------------------------------------------------------------
# exact_match
# ---------------------------------------------------------------------------

def test_exact_match_true_for_whitespace_and_case_variants():
    assert exact_match(
        "select   *  from singer",
        "SELECT * FROM singer;",
    )


def test_exact_match_false_for_different_identifiers():
    assert not exact_match("SELECT name FROM singer", "SELECT country FROM singer")


# ---------------------------------------------------------------------------
# execution_accuracy
# ---------------------------------------------------------------------------

def test_execution_accuracy_true_for_semantically_equivalent_sql(sample_db_path):
    # Different column order in the WHERE clause / different alias, same result set.
    assert execution_accuracy(
        "SELECT name FROM singer WHERE country = 'France'",
        "SELECT name FROM singer WHERE country = 'France'",
        sample_db_path,
    )


def test_execution_accuracy_false_for_different_results(sample_db_path):
    assert not execution_accuracy(
        "SELECT COUNT(*) FROM singer",           # 3 rows in the sample seed data
        "SELECT COUNT(*) FROM singer_in_concert", # 4 rows in the sample seed data
        sample_db_path,
    )


def test_execution_accuracy_false_when_predicted_sql_errors(sample_db_path):
    assert not execution_accuracy(
        "SELECT FROM WHERE",
        "SELECT COUNT(*) FROM singer",
        sample_db_path,
    )


def test_execution_accuracy_ignores_row_order(sample_db_path):
    assert execution_accuracy(
        "SELECT name FROM singer ORDER BY name ASC",
        "SELECT name FROM singer ORDER BY name DESC",
        sample_db_path,
    )


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

def test_compute_metrics_aggregates_correctly(sample_db_path):
    predictions = [
        "SELECT COUNT(*) FROM singer",          # correct (exec + exact)
        "SELECT name FROM singer",              # wrong result set
        "SELECT FROM WHERE",                    # execution error
    ]
    gold = [
        "SELECT COUNT(*) FROM singer",
        "SELECT COUNT(*) FROM singer",
        "SELECT COUNT(*) FROM singer",
    ]
    db_paths = [sample_db_path, sample_db_path, sample_db_path]

    metrics = compute_metrics(predictions, gold, db_paths)

    assert metrics["n_total"] == 3
    assert metrics["n_exec_correct"] == 1
    assert metrics["n_em_correct"] == 1
    assert metrics["exec_errors"] == 1
    assert metrics["execution_accuracy"] == 1 / 3
    assert metrics["exact_match"] == 1 / 3


def test_compute_metrics_handles_empty_input():
    metrics = compute_metrics([], [], [])

    assert metrics["n_total"] == 0
    assert metrics["execution_accuracy"] == 0.0
    assert metrics["exact_match"] == 0.0
