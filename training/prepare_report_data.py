"""Create a small, fictional report-format dataset; never reads user documents."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Match the actual RAG system prompt and message layout.
from chat import SYSTEM_PROMPT, NOT_FOUND

REPORT_REQUEST = (
    "Write a concise Markdown report using exactly these headings in this order: "
    "## Summary, ## Findings, ## Next steps. Cite factual claims with excerpt numbers. "
    "Clearly mark any proposed action as a recommendation."
)

# Each entry contains independent fictional evidence, a summary, findings, and a recommendation.
TRAIN = [
    ("Release Orion completed 18 of 20 tasks. Two accessibility checks remain open. The release review is Thursday.",
     "Orion has completed 18 of 20 tasks. [1]", "Two accessibility checks remain open. [1]\n- The release review is Thursday. [1]",
     "Recommendation: resolve the remaining accessibility checks before the review."),
    ("The support team received 75 tickets this week and closed 68. Seven tickets remain open. The oldest open ticket is three days old.",
     "Support closed 68 of 75 tickets received this week. [1]", "Seven tickets remain open. [1]\n- The oldest open ticket is three days old. [1]",
     "Recommendation: review the oldest open ticket first."),
    ("The office migration has moved 40 of 60 desks. Network testing is scheduled for Monday. Procurement has not confirmed delivery of the remaining monitors.",
     "The migration has moved 40 of 60 desks. [1]", "Network testing is scheduled for Monday. [1]\n- Delivery of the remaining monitors is unconfirmed. [1]",
     "Recommendation: confirm monitor delivery with procurement."),
    ("This month's infrastructure allocation is INR 90,000. Recorded spending is INR 72,000. The team will review storage usage on Wednesday.",
     "Recorded infrastructure spending is INR 72,000 against an INR 90,000 allocation. [1]", "The team will review storage usage on Wednesday. [1]",
     "Recommendation: use the storage review to identify possible savings."),
    ("Eight new employees joined this month. Six have completed security orientation. Two still need to attend the Friday session.",
     "Six of eight new employees have completed security orientation. [1]", "Two employees still need to attend the Friday session. [1]",
     "Recommendation: remind the remaining employees about the Friday session."),
    ("The inventory audit counted 125 laptops. Four have damaged screens. Replacement requests have been filed for all four.",
     "The inventory audit counted 125 laptops. [1]", "Four laptops have damaged screens. [1]\n- Replacement requests have been filed for all four. [1]",
     "Recommendation: track the four replacement requests until completion."),
    ("The analytics dashboard passed 34 of 36 acceptance checks. The two failures affect CSV export. A fix is assigned to the reporting team.",
     "The dashboard passed 34 of 36 acceptance checks. [1]", "Two failures affect CSV export. [1]\n- The reporting team is assigned the fix. [1]",
     "Recommendation: retest CSV export after the reporting team's fix."),
    ("The training workshop has 24 registered participants. The room holds 30 people. Materials will be distributed on Tuesday.",
     "The workshop has 24 registered participants. [1]", "The room holds 30 people. [1]\n- Materials will be distributed on Tuesday. [1]",
     "Recommendation: confirm the materials are ready before Tuesday."),
    ("The backup verification checked 12 archives. All 12 were restored successfully. The next verification is scheduled for April 12.",
     "All 12 checked archives were restored successfully. [1]", "The next verification is scheduled for April 12. [1]",
     "Recommendation: retain the verification results for the next review."),
    ("The documentation review covered 15 guides. Three contain outdated screenshots. The documentation owner will refresh them next week.",
     "The review covered 15 guides. [1]", "Three guides contain outdated screenshots. [1]\n- The documentation owner will refresh them next week. [1]",
     "Recommendation: verify the updated screenshots after the refresh."),
]
EVAL = [
    ("Project Cedar migrated 7 of 9 services. Two services await database access. The access request is assigned to the platform team.",
     "Cedar has migrated 7 of 9 services. [1]", "Two services await database access. [1]\n- The platform team owns the access request. [1]",
     "Recommendation: follow up with the platform team on database access."),
    ("The customer survey collected 52 responses. Eleven mention slow onboarding. A review meeting is scheduled for May 6.",
     "The survey collected 52 responses. [1]", "Eleven responses mention slow onboarding. [1]\n- A review meeting is scheduled for May 6. [1]",
     "Recommendation: review the onboarding feedback during the meeting."),
    ("The delivery review covered 28 shipments. Five arrived late. The carrier has been asked to provide an explanation by Friday.",
     "The delivery review covered 28 shipments. [1]", "Five shipments arrived late. [1]\n- The carrier has been asked for an explanation by Friday. [1]",
     "Recommendation: review the carrier's explanation when received."),
]


def example(evidence, summary, findings, action):
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "<document_excerpts>\n[1] demo-notes.txt\n" + evidence
         + "\n</document_excerpts>\n\nQuestion: Summarize the supplied notes. " + REPORT_REQUEST},
        {"role": "assistant", "content": f"## Summary\n{summary}\n\n## Findings\n- {findings}\n\n## Next steps\n- {action}"},
    ]}


def missing_example(evidence, question):
    row = example(evidence, "", "", "")
    row["messages"][1]["content"] = row["messages"][1]["content"].replace(
        "Summarize the supplied notes.", question)
    row["messages"][-1]["content"] = NOT_FOUND
    return row


def main():
    directory = ROOT / "training" / "data"
    directory.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": [example(*record) for record in TRAIN] + [
            missing_example("The workshop begins at 10 AM.", "What is the workshop's registration fee?"),
            missing_example("The audit found three outdated guides.", "Who is the audit director?"),
        ],
        "eval": [example(*record) for record in EVAL] + [
            missing_example("The maintenance window is Saturday.", "What is the server's purchase price?"),
        ],
    }
    for split, rows in splits.items():
        path = directory / f"report_{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        print(f"Wrote {len(rows)} fictional examples to {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
