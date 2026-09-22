"""Restore the rule numbers the scrape flattened out of the TPC policy pages.

The live pages (placement.iitbhu.ac.in/policies/students) are ordered lists,
and the policy text cross-references its own rules by number ('barring
exceptions as are detailed in Rule 4', 'the Rule 28 of the present policy will
apply'). The scraped markdown lost the numbering, so those references pointed
nowhere. This labels each paragraph '**Rule N.**' (sub-paragraphs
'**(Rule N, contd.)**'); src/agent/tools.py:resolve_rule_references uses the
labels to pull in a cited rule's full text. Numbering checked against the live
page's list and against every internal cross-reference.

Run once after (re-)scraping the policy pages, then re-run ingestion:
    python -m scripts.restore_policy_rule_numbers
    python -m src.ingest.run_ingestion

Each entry: (line_no, rule_no, contd, expected_opening_words). Aborts if any
line doesn't start with the expected words, so a label can't land on the
wrong paragraph."""
import sys
from pathlib import Path

PLACEMENT = [
 (12,1,0,"All full-time registered final year"),(14,2,0,"Applying to a company via"),
 (16,3,0,"A student can apply to a maximum of 50"),(18,4,0,"The Institute follows One-Student-One-Job"),
 (22,4,1,"A student who has secured a job"),(28,4,1,"A student can reapply for a maximum of 2"),
 (30,5,0,"There are cases where a company"),(32,6,0,"The final results of all the companies"),
 (34,7,0,"The students are requested to forward"),(36,8,0,"A student is allowed to appear for at most 4"),
 (40,9,0,"All the resumes are going to have"),(42,10,0,"The students will have to make entries"),
 (44,11,0,"Any data which is not supported"),(46,12,0,"Any person, who does not report"),
 (48,13,0,"No request for general data verification"),(50,14,0,"It may so happen that after the last date"),
 (54,15,0,"Any student receiving a Pre-Placement Offer"),(56,16,0,"Any student, found not to inform"),
 (58,17,0,"The students with PPO are required"),(60,18,0,"Campus selected interns will be allowed"),
 (62,19,0,"PPO will be counted as a job"),(64,20,0,"If PPO is accepted, the job will be recorded"),
 (68,21,0,"After the selection, the list of the companies"),(70,22,0,"In case the candidate, after accepting"),
 (74,23,0,"Students are forbidden from making direct contact"),(76,24,0,"The students facing any kind of problem"),
 (78,25,0,"Students are not allowed to carry mobile"),(80,26,0,"Any sort of misbehaviour/misconduct"),
 (82,27,0,"It is expected that a student shall NOT enter"),(84,28,0,"The students, who wish to withdraw"),
 (86,28,1,"If a student is found to be absent"),(88,28,1,"If a student is absent from the intended venue"),
 (90,29,0,"For any waiver from the punishment"),(92,30,0,"Students appearing for any test/GD/presentation"),
 (94,31,0,"Cheating or using unfair means"),(96,32,0,"Students must keep their Identity Card"),
 (98,33,0,"This Cell will ensure that placement activities"),(100,34,0,"Details about the company, having confirmed"),
 (102,35,0,"TPRs or TPVs will be judged"),(104,36,0,"Any other matter of indiscipline"),
]
INTERNSHIP = [
 (12,1,0,"All full-time registered students of B.Tech"),(14,2,0,"Applying for a company using the TPC portal"),
 (16,3,0,"The internship policy is 'One-Student-One-Internship'"),(18,4,0,"Final results of all the companies"),
 (20,5,0,"The students are requested to forward"),(22,6,0,"A student is allowed to appear for at most 2"),
 (24,7,0,"Any student willing to undergo a summer internship"),(28,8,0,"In case a student wants to reject the offer"),
 (30,9,0,"If the student rejects a paid internship offer"),(32,10,0,"If a student, having accepted a paid internship"),
 (36,11,0,"Students are forbidden from making direct contact"),(38,12,0,"The students facing any kind of problem"),
 (40,13,0,"It is expected that a student shall NOT enter"),(42,14,0,"Students are not allowed to carry mobile"),
 (44,15,0,"Any sort of misbehaviour/misconduct"),(46,16,0,"The students, who wish to withdraw"),
 (48,17,0,"Students appearing for any test/GD/presentation"),(50,19,0,"Cheating or using unfair means"),
 (52,20,0,"Students must keep their Identity Card"),(54,21,0,"Details about the company, having confirmed"),
 (56,22,0,"TPRs or TPVs will be judged"),(58,23,0,"Any other matter of indiscipline"),
 (62,24,0,"All the resumes are going to have"),(64,25,0,"The students will have to make entries"),
 (66,26,0,"Any data which is not supported"),(68,27,0,"Any person, who does not report"),
 (70,28,0,"No request for general data verification"),(72,29,0,"It may so happen that after the last date"),
]
NOTE = ("> Rule numbers (**Rule N.**) restored from the numbered list on the live page -- "
        "the original scrape flattened the page's ordered list, which left the policy's "
        "own cross-references ('Rule 4', 'Rule 28', ...) unresolvable. "
        "'(Rule N, contd.)' marks a paragraph that is a sub-part of rule N.")

def label(path, table):
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    if any("**Rule " in l for l in lines):
        sys.exit(f"{path}: already labelled, refusing to double-label")
    for ln, n, contd, expect in table:
        line = lines[ln - 1]
        if not line.startswith(expect):
            sys.exit(f"{path}:{ln}: expected {expect!r}, found {line[:60]!r}")
        lines[ln - 1] = (f"**(Rule {n}, contd.)** " if contd else f"**Rule {n}.** ") + line
    # note goes right after the 'Fetched:' line
    fi = next(i for i, l in enumerate(lines) if l.startswith("Fetched:"))
    lines.insert(fi + 1, "")
    lines.insert(fi + 2, NOTE)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"labelled {len(table)} paragraphs in {Path(path).name}")

base = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent.parent / "data" / "raw" / "policy")
label(f"{base}/student_placement_rules_and_regulations.md", PLACEMENT)
label(f"{base}/student_internship_rules_and_regulations.md", INTERNSHIP)
