from autoprep.pipeline import AutoPrepPipeline
from autoprep.interactor import HumanPrompter
import json

if __name__ == "__main__":
    # ── Non-interactive mode (default - runs automatically) ──
    pipeline = AutoPrepPipeline(
        missing_strategy="auto",
        outlier_method="iqr",
        outlier_action="none",  # Changed from "clip" to preserve data by default
        encoding_strategy="auto",
        extract_date_features=True,
        visualize=False,
        output_dir="reports/figures",
        interactive_mode=False,  # Set to True for human-in-the-loop mode (see below)
    )

    # ── To use INTERACTIVE MODE (terminal prompts): ──
    # Uncomment the code block below and comment out the above
    # This will prompt you for decisions on Yellow zones and high cardinality columns
    # NOTE: Interactive mode ONLY works in terminal, not in Streamlit!
    """
    prompter = HumanPrompter()
    pipeline = AutoPrepPipeline(
        missing_strategy="auto",
        outlier_method="iqr",
        outlier_action="none",
        encoding_strategy="auto",
        extract_date_features=True,
        visualize=False,
        output_dir="reports/figures",
        interactive_mode=True,
        human_prompter=prompter,
    )
    """

    df, report = pipeline.run_and_save(
        "data/sample.csv",
        output_csv="reports/processed_data.csv",
        report_json="reports/report.json",
    )

    print("\n── Processed DataFrame ──")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print("\n── Cleaning Summary ──")
    print(json.dumps(report["cleaning"], indent=2, default=str))

    print("\n── Encoding Summary ──")
    print(json.dumps(report["encoding"], indent=2, default=str))

    print("\n── Feature Engineering Summary ──")
    print(json.dumps(report["feature_engineering"], indent=2, default=str))

    print(f"\n── Figures saved ({len(report['figures'])}) ──")
    for fig in report["figures"]:
        print(" ", fig)


