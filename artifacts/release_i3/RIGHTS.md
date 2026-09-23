# Release component rights and attribution (local review)

Date: 2026-09-23. This records publisher-declared terms, not a legal opinion or independent relicensing of embedded third-party material.

| Component | Pinned source and retained evidence | Declared terms | Remaining uncertainty |
|---|---|---|---|
| Laya code | `data/laya@d113dca2512fb3eaca313534bc54c7162d87c1d4`, `data/laya/LICENSE` SHA-256 `a6cba85bc92e0cff7a450b1d873c0eaa2e9fc96bf472df0247a26bec77bf3ff9` | Apache-2.0 source license, copied into bundle | Retain license, modification/attribution notices on redistribution |
| Laya base weights and tokenizer | `convaiinnovations/laya@1c5edc17a7acd8701df6fc341c0d179f1c62c982`; pinned snapshot README SHA-256 `33911629d87484754bb755d2118a38626264a9b63bc33ee30131d7564f5560ee` | Snapshot model-card YAML declares `license: apache-2.0` | No separate weight/tokenizer LICENSE in local snapshot; publisher metadata is the available declaration |
| Training trace dataset | `11-47/glm-5.2-coding-and-debugging-traces@1371ed38f8890d0520a53bc7ad850308eb4d7a22`; downloaded pinned README to ignored `data/i3_source_dataset_README.md`, SHA-256 `c4c15b9a067d478f9a4f23a8078c58755dff04b01af0878e2b55037542dc333b`; source URL below | README YAML says `license: cc-by-4.0` and text explicitly permits training/derivatives with attribution | Dataset publisher's authority over embedded third-party repositories, tasks and quoted content is not independently attested; source teacher/real-execution claims are publisher-attributed |
| System-A inference adaptation | `release/i3/` and hash-pinned local bundle | New code and derived checkpoint prepared for review | No explicit System-A repository-level license identified for the new code; choose it before public GitHub distribution |

Pinned dataset README URL: `https://huggingface.co/datasets/11-47/glm-5.2-coding-and-debugging-traces/raw/1371ed38f8890d0520a53bc7ad850308eb4d7a22/README.md`. Retrieved with `curl --fail --location --silent --show-error <URL> --output data/i3_source_dataset_README.md`, then `sha256sum` inside the repository. No paid search/scraping service was used.

Proposed visible attribution in any eventual public card: **Laya** (`convaiinnovations/laya`, pinned model/source revisions) and **GLM 5.2 Agent Traces** (`11-47/glm-5.2-coding-and-debugging-traces`, pinned source revision; publisher's README suggests attribution to `greghavens/glm-5.2-coding-and-debugging-traces`). Keep both names and the dataset URL rather than silently replacing one with the other. No raw trace rows are included in the inference bundle. Final public upload still requires owner review of the embedded-content and code-license questions.
