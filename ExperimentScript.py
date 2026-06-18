from __future__ import annotations
import random
import traceback
from collections import Counter
from api_examples_common import (
    get_json,
    parse_args,
    post_json,
    print_export_summary,
    print_response_checks,
    print_section,
)
import json
from pathlib import Path
import random
from collections import Counter

###     THE RANDOM SPLIT USED FOR THE ACTUAL EXPERIMENT
# THIS RESULTED IN ONLY 1/4TH OF THE PARTICIPANTS USABLE FOR THE ANALYSIS
#
# A = [f"{i}" for i in range(80)]
#
# n_lists = 500
# list_size = 5
#
# total_slots = n_lists * list_size
#
# base_count = total_slots // len(A)
# extra = total_slots % len(A)
#
# counts = {a: base_count for a in A}
#
# # 40 random elements get one extra appearance
# for a in random.sample(A, extra):
#     counts[a] += 1
#
# # Create empty lists
# B = [[] for _ in range(n_lists)]
# remaining_capacity = [list_size] * n_lists
#
# # Place items with highest remaining count first
# items = list(A)
# random.shuffle(items)
#
# for item in sorted(items, key=lambda x: counts[x], reverse=True):
#     k = counts[item]
#
#     # choose k distinct lists with the most remaining capacity
#     available = [i for i in range(n_lists) if remaining_capacity[i] > 0]
#
#     chosen_lists = random.sample(
#         sorted(available, key=lambda i: remaining_capacity[i], reverse=True)[:k+10],
#         k
#     )
#
#     for idx in chosen_lists:
#         B[idx].append(item)
#         remaining_capacity[idx] -= 1
#
# # Verify
# assert all(len(lst) == list_size for lst in B)
# assert all(len(lst) == len(set(lst)) for lst in B)
#
# flat = [x for lst in B for x in lst]
# freq = Counter(flat)
#
# print("min appearances =", min(freq.values()))
# print("max appearances =", max(freq.values()))
#
# # Shuffle within each list
# instance_counter = {}
# for lst in B:
#     random.shuffle(lst)
#     for i in lst:
#         if i in instance_counter:
#             instance_counter[i] += 1
#         else:
#             instance_counter[i] = 1
#
# print(instance_counter)
# print(min(counts.values()))  # 121
# print(max(counts.values()))  # 122
# print(sum(counts.values()))  # 9720
#
#
#
#
human_true = []
human_false = []
ai_true = []
ai_false = []

dataset_file = open("Dataset for survey - Experiment dataset.tsv", "r")

dataset_file.readline()  # skip first row
for line in dataset_file.readlines():
    split_line = line.split("\t")
    human_true.append(split_line[1])
    human_false.append(split_line[2])
    ai_true.append(split_line[3])
    ai_false.append(split_line[4].strip("\n"))

dataset_file.close()

dataset = human_true + human_false + ai_true + ai_false

statements_base = [
    {
        "statement_id": f"TS{count}",
        "statement_text": text,
        "known_truth": count in range(0, 20) or count in range(40, 60),
        "actual_source": "human" if count in range(0, 40) else "ai",
    }
    for count, text in enumerate(dataset)
]
#
# # print(statements_base)
#
# statements = []
# for i in range(len(B)):
#     statements.append([])
#     for j in B[i]:
#         statements[i].append(statements_base[int(j)])


n_personas = 500

# =====================================================
# Split statement IDs into the 4 required conditions
# =====================================================

human_true_ids = list(range(0, 20))
human_false_ids = list(range(20, 40))
ai_true_ids = list(range(40, 60))
ai_false_ids = list(range(60, 80))


### THE BALANCED DATASET SPLIT - THIS WOULD RESULT IN ALL 500 PARTICIPANTS SEEING EACH OF THE CONDITIONS

def balanced_pool(ids):
    """
    Each statement appears exactly equal times across personas.
    """
    repeats = n_personas // len(ids)  # 25
    pool = ids * repeats

    remainder = n_personas % len(ids)
    if remainder:
        pool.extend(random.sample(ids, remainder))

    random.shuffle(pool)
    return pool


human_true_pool = balanced_pool(human_true_ids)
human_false_pool = balanced_pool(human_false_ids)
ai_true_pool = balanced_pool(ai_true_ids)
ai_false_pool = balanced_pool(ai_false_ids)

B = []

for i in range(n_personas):
    participant_set = [
        human_true_pool[i],
        human_false_pool[i],
        ai_true_pool[i],
        ai_false_pool[i]
    ]

    random.shuffle(participant_set)

    B.append(participant_set)

statements = []

for i in range(len(B)):
    statements.append([])

    for j in B[i]:
        statements[i].append(statements_base[int(j)])

flat = [x for lst in B for x in lst]
freq = Counter(flat)

print("min appearances =", min(freq.values()))
print("max appearances =", max(freq.values()))

print(B)

# Create output directory once
Path("exports").mkdir(exist_ok=True)

PRE_SURVEY_MESSAGE = """
Please answer the following pre-study questions. Return JSON only.

Questions:
1. ai_familiarity_1_to_7: How familiar are you with AI-generated text? 1 = not at all familiar, 7 = extremely familiar.
2. ai_trust_1_to_7: In general, how much do you trust AI systems to provide accurate information? 1 = do not trust at all, 7 = trust completely.
3. online_information_skepticism_1_to_7: How skeptical are you of factual claims you see online? 1 = not skeptical at all, 7 = extremely skeptical.
4. self_rated_fact_checking_frequency_1_to_7: How often do you fact-check information before believing or sharing it? 1 = never, 7 = always.
5. baseline_confidence_in_truth_judgments_1_to_7: How confident are you in your ability to judge whether short factual statements are true or false? 1 = not confident at all, 7 = extremely confident.
""".strip()

POST_SURVEY_MESSAGE = """
Please answer the following post-study questions based on the statement-judgment task you just completed. Return JSON only.

Questions:
1. perceived_task_difficulty_1_to_7: Overall, how difficult was it to judge the statements? 1 = very easy, 7 = very difficult.
2. perceived_accuracy_1_to_7: How accurate do you think your judgments were? 1 = not accurate at all, 7 = extremely accurate.
3. confidence_change_minus3_to_3: Compared with the start of the task, how did your confidence change? -3 = much less confident, 0 = no change, 3 = much more confident.
4. relied_on_source_label_1_to_7: How much did source information, if shown, affect your judgments? 1 = not at all, 7 = a great deal.
5. open_ended_strategy: In one sentence, what was your main strategy for judging the statements?
""".strip()


def main() -> None:
    args = parse_args("Run truth/source labeled and unlabeled API examples.")

    # assert args.personas == 500
    # assert len(statements) == 500
    for persona_index in range(1, args.personas + 1):
        try:
            if persona_index <= args.personas // 2:
                setup_id = "truth_source_unlabeled"
            else:
                setup_id = "truth_source_labeled"
            session_payload = {
                "experiment_setup_id": setup_id,
                "study": {
                    "name": "Statement judgement task",
                    "description": "Participants judge truthfulness and trustworthiness.",
                    "instructions": "Answer each item using only the requested JSON fields.",
                },
                "criteria": {"age": {"min": 18, "max": 25}},
            }
            print_section(f"CREATE SESSION PAYLOAD ({setup_id}, persona {persona_index}/{args.personas})",
                          session_payload)
            session = post_json(args.base_url, "/v1/sessions", session_payload)
            print_section("CREATE SESSION RESPONSE", session)

            pre_payload = {
                "message": PRE_SURVEY_MESSAGE,
                "stimulus": {"questionnaire_id": "truth_source_pre_survey"},
                "metadata": {
                    "example_script": "run_truth_source_example.py",
                    "phase": "pre_survey",
                },
                "trial_id": "pre_survey",
                "trial_index": 0,
                "reset_policy": "carryover",
                "response_mode": "survey",
                "capture_thinking": True,
            }
            print_section("PRE-SURVEY PAYLOAD", pre_payload)
            pre = post_json(args.base_url, f"/v1/sessions/{session['session_id']}/turns", pre_payload)
            print_section("PRE-SURVEY RESPONSE", pre)
            pre_fields = [
                "ai_familiarity_1_to_7",
                "ai_trust_1_to_7",
                "online_information_skepticism_1_to_7",
                "self_rated_fact_checking_frequency_1_to_7",
                "baseline_confidence_in_truth_judgments_1_to_7",
            ]
            print_response_checks(pre, required_fields=pre_fields, rating_fields=pre_fields)

            participant_statements = statements[persona_index - 1]
            for trial_index, statement in enumerate(participant_statements, start=1):
                labeled = setup_id == "truth_source_labeled"
                if labeled:
                    message = (
                        f"Please evaluate this statement.\n\n"
                        f"Statement: {statement['statement_text']}\n"
                        f"Source label shown to you: {statement['actual_source']}-created.\n\n"
                        "Return JSON only with answers to these exact questions:\n"
                        "1. predicted_truthfulness: Is the statement true or false? Use true, false, or unsure.\n"
                        "2. confidence_1_to_7: How confident are you in that judgment? 1 = not at all confident, 7 = extremely confident.\n"
                        "3. trustworthiness_1_to_7: How trustworthy does this statement seem? 1 = not trustworthy at all, 7 = extremely trustworthy.\n"
                    )
                else:
                    message = (
                        f"Please evaluate this statement.\n\n"
                        f"Statement: {statement['statement_text']}\n"
                        "No source label is shown.\n\n"
                        "Return JSON only with answers to these exact questions:\n"
                        "1. predicted_truthfulness: Is the statement true or false? Use true, false, or unsure.\n"
                        "2. confidence_1_to_7: How confident are you in that judgment? 1 = not at all confident, 7 = extremely confident.\n"
                        "3. perceived_source_1_human_to_7_ai: Who do you think probably created this statement? 1 = definitely human, 7 = definitely AI.\n"
                        "4. trustworthiness_1_to_7: How trustworthy does this statement seem? 1 = not trustworthy at all, 7 = extremely trustworthy.\n"
                    )

                stimulus = {
                    **statement,
                    "label_visible": labeled,
                }
                if labeled:
                    stimulus["shown_source_label"] = f"{statement['actual_source']}-created"

                turn_payload = {
                    "message": message,
                    "stimulus": stimulus,
                    "metadata": {
                        "example_script": "run_truth_source_example.py",
                        "phase": "statement_task",
                        "condition_hidden_from_message": not labeled,
                    },
                    "trial_id": statement["statement_id"],
                    "trial_index": trial_index,
                    "reset_policy": "carryover",
                    "response_mode": "experiment",
                    "capture_thinking": True,
                }
                print_section(
                    f"STATEMENT TURN PAYLOAD, STATEMENT {trial_index}/5, PERSONA {persona_index}/{args.personas}",
                    turn_payload)
                turn = post_json(args.base_url, f"/v1/sessions/{session['session_id']}/turns", turn_payload)
                print_section("STATEMENT TURN RESPONSE", turn)
                required_fields = [
                    "predicted_truthfulness",
                    "confidence_1_to_7",
                    "trustworthiness_1_to_7",
                ]
                rating_fields = ["confidence_1_to_7", "trustworthiness_1_to_7"]
                if not labeled:
                    required_fields.append("perceived_source_1_human_to_7_ai")
                    rating_fields.append("perceived_source_1_human_to_7_ai")
                print_response_checks(
                    turn,
                    required_fields=required_fields,
                    rating_fields=rating_fields,
                )

            post_payload = {
                "message": POST_SURVEY_MESSAGE,
                "stimulus": {"questionnaire_id": "truth_source_post_survey"},
                "metadata": {
                    "example_script": "run_truth_source_example.py",
                    "phase": "post_survey",
                },
                "trial_id": "post_survey",
                "trial_index": len(participant_statements) + 1,
                "reset_policy": "carryover",
                "response_mode": "survey",
                "capture_thinking": True,
            }
            print_section("POST-SURVEY PAYLOAD", post_payload)
            post = post_json(args.base_url, f"/v1/sessions/{session['session_id']}/turns", post_payload)
            print_section("POST-SURVEY RESPONSE", post)
            post_fields = [
                "perceived_task_difficulty_1_to_7",
                "perceived_accuracy_1_to_7",
                "confidence_change_minus3_to_3",
                "relied_on_source_label_1_to_7",
                "open_ended_strategy",
            ]
            print_response_checks(
                post,
                required_fields=post_fields,
                rating_fields=[
                    "perceived_task_difficulty_1_to_7",
                    "perceived_accuracy_1_to_7",
                    "relied_on_source_label_1_to_7",
                ],
            )

            export = get_json(args.base_url, f"/v1/sessions/{session['session_id']}/export")
            print_export_summary(export)

            output = {
                "setup_id": setup_id,
                "persona_index": persona_index,
                "session_id": session["session_id"],

                # Experimental condition
                "condition": {
                    "source_label_visible": setup_id == "truth_source_labeled"
                },

                # Statements shown to this participant
                "statements": participant_statements,

                # Full API export
                "export": export
            }

            filename = (
                f"exports/"
                f"{setup_id}_persona_{persona_index:03d}.json"
            )

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

            print(f"Saved {filename}")
        except Exception as e:
            with open(str(persona_index) + "ERROR", "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
            print(f"ERROR FOR PERSONA {persona_index}: {traceback.format_exc()}")


if __name__ == "__main__":
    print("Run main")
    main()
