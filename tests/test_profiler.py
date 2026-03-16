import pandas as pd

from autoprep.profiler import DataProfiler


def test_assess_missing_data_traffic_light_tags():
    df = pd.DataFrame(
        {
            "green_col": [1, 2, 3, 4, 5, 6, 7, 8, 9, None],
            "yellow_col": [1, 2, 3, None, None, None, None, None, None, None],
            "red_col": [None, None, None, None, None, None, None, None, None, 1],
        }
    )

    profiler = DataProfiler(
        missing_green_zone=(0.0, 0.10),
        missing_yellow_zone=(0.11, 0.79),
        missing_red_zone=(0.80, 1.0),
    )
    report = profiler.assess_missing_data(df)

    assert report["green_col"]["traffic_light"] == "Green"
    assert report["yellow_col"]["traffic_light"] == "Yellow"
    assert report["red_col"]["traffic_light"] == "Red"


def test_assess_cardinality_limit_flag():
    df = pd.DataFrame(
        {
            "city": ["a", "b", "a", "c"],
            "id_str": ["u1", "u2", "u3", "u4"],
        }
    )

    profiler = DataProfiler(cardinality_limit=3)
    report = profiler.assess_cardinality(df)

    assert report["city"]["ask_human"] is False
    assert report["id_str"]["ask_human"] is True


def test_generate_health_report_has_required_sections():
    df = pd.DataFrame(
        {
            "num": [1, 2, 3, 100, 5],
            "cat": ["x", "y", "y", "z", "x"],
        }
    )

    profiler = DataProfiler()
    report = profiler.generate_health_report(df)

    assert "shape" in report
    assert "missing_data" in report
    assert "outliers" in report
    assert "cardinality" in report
