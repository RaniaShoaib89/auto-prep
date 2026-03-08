from autoprep.pipeline import AutoPrepPipeline
import json

if __name__ == "__main__":
    pipeline = AutoPrepPipeline(
        missing_strategy="auto",
        outlier_method="iqr",
        outlier_action="clip",
        encoding_strategy="auto",
        extract_date_features=True,
        visualize=True,
        output_dir="reports/figures",
    )

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

