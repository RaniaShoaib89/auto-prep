from autoprep.interactor import HumanPrompter


def test_parse_report_separates_auto_and_human_tasks():
    health_report = {
        "missing_data": {
            "green_col": {"missing_pct": 5.0, "traffic_light": "Green"},
            "yellow_col": {"missing_pct": 25.0, "traffic_light": "Yellow"},
            "red_col": {"missing_pct": 90.0, "traffic_light": "Red"},
        },
        "cardinality": {
            "city": {"n_unique": 12, "ask_human": False},
            "user_id": {"n_unique": 5000, "ask_human": True},
        },
    }

    prompter = HumanPrompter(input_fn=lambda _: "2")
    result = prompter.parse_report(health_report)

    assert result["auto_tasks"]["missing"]["green_col"]["action"] == "basic_impute"
    assert result["auto_tasks"]["missing"]["red_col"]["action"] == "drop_column"
    assert "yellow_col" in result["human_tasks"]["missing"]
    assert "user_id" in result["human_tasks"]["cardinality"]


def test_prompt_methods_capture_choices():
    answers = iter(["3", "2"])
    prompter = HumanPrompter(input_fn=lambda _: next(answers))

    missing_decision = prompter.prompt_missing_yellow_zone("age", 34.56)
    card_decision = prompter.prompt_high_cardinality("city", 88)

    assert missing_decision["action"] == "smart_knn_impute"
    assert card_decision["action"] == "keep_top_10"


def test_build_action_plan_combines_auto_and_human_decisions():
    health_report = {
        "missing_data": {
            "green_col": {"missing_pct": 7.0, "traffic_light": "Green"},
            "yellow_col": {"missing_pct": 33.0, "traffic_light": "Yellow"},
            "red_col": {"missing_pct": 88.0, "traffic_light": "Red"},
        },
        "cardinality": {
            "city": {"n_unique": 15, "ask_human": False},
            "user_id": {"n_unique": 3000, "ask_human": True},
        },
    }

    answers = iter(["2", "1"])
    prompter = HumanPrompter(input_fn=lambda _: next(answers))
    prompter.parse_report(health_report)
    prompter.prompt_missing_yellow_zone("yellow_col", 33.0)
    prompter.prompt_high_cardinality("user_id", 3000)

    plan = prompter.build_action_plan()

    assert plan["actions"]["missing"]["green_col"] == "basic_impute"
    assert plan["actions"]["missing"]["yellow_col"] == "basic_impute"
    assert plan["actions"]["missing"]["red_col"] == "drop_column"
    assert plan["actions"]["cardinality"]["city"] == "auto_encode"
    assert plan["actions"]["cardinality"]["user_id"] == "drop_column"

    assert "red_col" in plan["cleaner_instructions"]["drop_columns"]
    assert "user_id" in plan["cleaner_instructions"]["drop_columns"]
    assert plan["cleaner_instructions"]["missing_imputation"]["yellow_col"] == "basic_impute"
