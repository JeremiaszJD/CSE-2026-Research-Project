import json
from pathlib import Path

import numpy as np
import pandas as pd

##############################################################################
# CONFIG
##############################################################################

EXPORT_DIR = Path("exports")


##############################################################################
# HELPERS
##############################################################################

def safe_json_load(value):
    """
    Some response fields are JSON strings.
    Safely parse them.
    """
    if isinstance(value, dict):
        return value

    if not value:
        return {}

    try:
        return json.loads(value)
    except Exception:
        return {}


##############################################################################
# PARSE EXPORTS
##############################################################################

rows = []

files = list(EXPORT_DIR.glob("*.json"))

print(f"Found {len(files)} export files")

for file in files:

    try:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        participant_id = data.get("persona_index")

        label_visible = int(
            data.get("condition", {})
            .get("source_label_visible", False)
        )

        export = data.get("export", {})

        ############################################################
        # PERSONA VARIABLES
        ############################################################

        persona = export.get("persona", {})

        ############################################################
        # PRE-SURVEY VARIABLES
        ############################################################

        pre_survey = {}

        turns = export.get("turns", [])

        for turn in turns:

            if turn.get("trial_id") == "pre_survey":
                pre_survey = safe_json_load(
                    turn.get("response")
                )

                break

        ############################################################
        # STATEMENT TRIALS
        ############################################################

        for turn in turns:

            trial_id = str(
                turn.get("trial_id", "")
            )

            # Skip surveys
            if not trial_id.startswith("TS"):
                continue

            stimulus = turn.get("stimulus", {})

            response = safe_json_load(
                turn.get("response")
            )

            predicted_truth = (
                str(
                    response.get(
                        "predicted_truthfulness",
                        ""
                    )
                )
                .strip()
                .lower()
            )

            known_truth = stimulus.get(
                "known_truth"
            )

            ########################################################
            # ACCURACY
            ########################################################

            correct = np.nan

            if predicted_truth in ["true", "false"]:
                predicted_bool = (
                        predicted_truth == "true"
                )

                correct = int(
                    predicted_bool == known_truth
                )

            ########################################################
            # ROW
            ########################################################

            rows.append({

                ####################################################
                # IDs
                ####################################################

                "participant_id":
                    participant_id,

                "statement_id":
                    stimulus.get(
                        "statement_id"
                    ),

                ####################################################
                # EXPERIMENTAL FACTORS
                ####################################################

                "label_visible":
                    label_visible,

                "actual_source":
                    stimulus.get(
                        "actual_source"
                    ),

                "known_truth":
                    known_truth,

                ####################################################
                # RESPONSES
                ####################################################

                "predicted_truthfulness":
                    predicted_truth,

                "correct":
                    correct,

                "confidence":
                    response.get(
                        "confidence_1_to_7"
                    ),

                "trustworthiness":
                    response.get(
                        "trustworthiness_1_to_7"
                    ),

                "perceived_source":
                    response.get(
                        "perceived_source_1_human_to_7_ai"
                    ),

                ####################################################
                # PRE-SURVEY VARIABLES
                ####################################################

                "presurvey_ai_familiarity":
                    pre_survey.get(
                        "ai_familiarity_1_to_7"
                    ),

                "presurvey_ai_trust":
                    pre_survey.get(
                        "ai_trust_1_to_7"
                    ),

                "presurvey_skepticism":
                    pre_survey.get(
                        "online_information_skepticism_1_to_7"
                    ),

                "presurvey_fact_checking":
                    pre_survey.get(
                        "self_rated_fact_checking_frequency_1_to_7"
                    ),

                "presurvey_baseline_confidence":
                    pre_survey.get(
                        "baseline_confidence_in_truth_judgments_1_to_7"
                    ),

                ####################################################
                # PERSONA VARIABLES
                ####################################################

                "age":
                    persona.get("age"),

                "gender":
                    persona.get("gender"),

                "nationality":
                    persona.get("nationality"),

                "education":
                    persona.get("education"),

                ####################################################
                # PERSONA AI VARIABLES
                ####################################################

                "persona_ai_familiarity":
                    persona.get("ai_familiarity"),

                "persona_ai_trust":
                    persona.get("ai_trust"),

                "persona_ai_literacy":
                    persona.get("ai_literacy"),

                ####################################################
                # PERSONA CONFIDENCE VARIABLES
                ####################################################

                "persona_general_confidence":
                    persona.get(
                        "general_confidence"
                    ),

                "persona_reasoning_style":
                    persona.get(
                        "reasoning_style"
                    ),

                "persona_online_content_skepticism":
                    persona.get(
                        "online_content_skepticism"
                    ),

                ####################################################
                # KNOWLEDGE VARIABLES
                ####################################################

                "knowledge_geography_history":
                    persona.get(
                        "knowledge_geography_history"
                    ),

                "knowledge_science_health":
                    persona.get(
                        "knowledge_science_health"
                    ),

                "knowledge_entertainment_literature":
                    persona.get(
                        "knowledge_entertainment_literature"
                    ),

                "knowledge_technology_internet":
                    persona.get(
                        "knowledge_technology_internet"
                    ),

                ####################################################
                # OPTIONAL EXTRA
                ####################################################

                "topic_familiarity":
                    persona.get(
                        "topic_familiarity"
                    ),

                ####################################################
                # FILE INFO
                ####################################################

                "source_file":
                    file.name
            })

    except Exception as e:

        print(
            f"Failed parsing {file.name}: {e}"
        )

##############################################################################
# CREATE DATAFRAME
##############################################################################

df = pd.DataFrame(rows)

##############################################################################
# BASIC CHECKS
##############################################################################

print("\n==============================")
print("DATASET SUMMARY")
print("==============================")

print(f"Rows: {len(df)}")
print(f"Participants: {df['participant_id'].nunique()}")
print(f"Statements: {df['statement_id'].nunique()}")

print("\nMissing values:")
print(df.isna().sum())

##############################################################################
# SAVE
##############################################################################

output_file = "analysis_dataset.csv"

df.to_csv(
    output_file,
    index=False
)

print(f"\nSaved {output_file}")

##############################################################################
# OPTIONAL QUICK DESCRIPTIVES
##############################################################################

print("\nAccuracy by source")

print(
    df.groupby("actual_source")["correct"]
    .mean()
)

print("\nAccuracy by condition")

print(
    df.groupby("label_visible")["correct"]
    .mean()
)

print("\nTrustworthiness by source")

print(
    df.groupby("actual_source")["trustworthiness"]
    .mean()
)
