import pandas as pd

from autoprep.interactor import HumanPrompter
from autoprep.pipeline import AutoPrepPipeline


def test_pipeline_interactive_mode_builds_and_applies_action_plan(tmp_path):
    rows = 60
    df = pd.DataFrame(
        {
            "num": list(range(rows)),
            "miss_yellow": [None if i < 18 else float(i) for i in range(rows)],
            "high_card": [f"id_{i}" for i in range(rows)],
        }
    )

    csv_path = tmp_path / "interactive_input.csv"
    df.to_csv(csv_path, index=False)

    answers = iter(["2", "1"])
    prompter = HumanPrompter(input_fn=lambda _: next(answers))

    pipeline = AutoPrepPipeline(
        visualize=False,
        interactive_mode=True,
        human_prompter=prompter,
        drop_identifiers=False,
    )

    processed_df, report = pipeline.run(str(csv_path))

    assert report["action_plan"] is not None
    assert report["action_plan"]["actions"]["missing"]["miss_yellow"] == "basic_impute"
    assert report["action_plan"]["actions"]["cardinality"]["high_card"] == "drop_column"
    assert "high_card" in report["action_plan"]["cleaner_instructions"]["drop_columns"]
    assert "high_card" not in processed_df.columns
