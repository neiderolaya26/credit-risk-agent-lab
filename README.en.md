# credit-risk-agent-lab · English summary

> v1.2.0 · An end-to-end credit decision system built only on public Lending Club data. The full documentation, notebooks and the 23 design decisions are in Spanish: [README.md](README.md) · [docs/decisiones.md](docs/decisiones.md).

**In 30 seconds:**

- **Risk that pays:** a LightGBM model trained only on public application data ranks risk better than Lending Club's own score. At the same 20% rejection rate it adds **USD 337k of realized profit per 10,000 applications** (95% CI [272k; 408k], out of time on 2015 loans). That is 0.32% of the USD 104M lent and 4.7% more profit than ranking by Lending Club's score. Measured on the policy the system actually applies and net of assumed costs, the policy loses USD 0.8M less than approving everything, but it does not beat a simple subgrade cut with the same rejection rate.
- **The whole decision, not just a score:** prepayment with a discrete-time survival model, economic loss, limit and pricing under a 20% payment-to-income cap, and a causal analysis of the effect of the rate on default, reported as a negative result.
- **AI agent with human control:** the decision is made by the credit policy in code. A LangGraph graph with three Claude specialists analyzes the case, an AI committee gives its opinion and writes the memo, and an auditor reviews it. Only cases the policy approves and the committee confirms go through on their own; everything else, including disagreements, is signed by an analyst. Reasons for the applicant come from a fixed catalog, not from the LLM.

**Try it:** [live demo of the engine](https://credit-risk-agent-lab.streamlit.app/) (Streamlit, no LLM) · [models and model card on Hugging Face](https://huggingface.co/neiderolaya26/credit-risk-lending-club)

![Extra profit of the champion model versus Lending Club's score](docs/img/valor_campeon.png)

## Key results (out of time, 2015)

| Piece | Result |
|---|---|
| Risk | Champion LightGBM AUC 0.6823 vs 0.6784 for Lending Club's score, using only public application variables; 95% CI of the difference [+0.0016; +0.0063] |
| Model value | Ranking by the champion instead of by Lending Club's rate adds USD 195k (10% rejection), 337k (20%) and 447k (30%) of realized profit per 10,000 applications. Default barely moves (11.7% vs 11.9% at 20%): the gain comes from not rejecting expensive but good loans. In context: at 20% about USD 104M is lent per 10,000 applications (mean loan USD 12,806), so +337k is 0.32% of the amount lent and 4.7% more profit than Lending Club's rule (USD 7.16M). The CI only reflects 2015 sampling noise, not a change in the credit cycle |
| Approval | An expected-utility policy matches the profit of a PD cutoff with a better ROI (6.30%) and lower default |
| Prepayment and loss | Discrete-time survival beats the alternatives (RMSE 0.0497); classic LGD has no signal |
| Limit and pricing | Loans Lending Club priced below the suggested rate returned 2.8% vs 6.7% for the rest |
| Real policy, net | Backtest on 10,000 applications (Jul–Dec 2015): the policy approves 84.9%, counteroffers 13.8% and rejects 1.3%. With dummy, illustrative costs (3% funding, USD 100 per loan, 10% capital at 12%), per 10,000 applications: Lending Club approving everything −USD 1.48M; the policy −0.69M to −1.11M depending on counteroffer acceptance; a subgrade cut with the same rejection rate −0.50M |
| Causal inference | The rate–default correlation is explained by subgrade; the causal effect of the rate **could not be identified**, because the placebo fails with every method (also with wild cluster bootstrap and two-way clustering, p < 0.001). The engine assumes no effect and adds a stress scenario, which is an assumption: with 0, 0.5, 1 or 2 pts of PD per point of overpricing, 0.1%, 0.4%, 1.6% or 2.2% of applications have no break-even rate |
| Agent | Claude Haiku 5.5 (version v3) on 35 cases with real outcomes: agrees with the policy in 100% of cases (v2: 80% on the same references), 0 of 35 committee drafts break a hard rule, 94% of memo figures backed by a tool (a random number would pass 12% of the time), 100% of audits passed on the first try and 98% on an Opus 5.5 judge, at USD 0.007 per application. In v4 the code decides: the committee's opinion matches the policy in 34 of 35 cases (97%, all 6 rejections included), 0 of 37 drafts break a rule, 94% fidelity and 94% first-try audits, at USD 0.008 per application. The one disagreement would go to an analyst, flagged. The v4 judge run is pending |

## Limits and fairness

- **The agent does not decide alone on cases that hurt the applicant.** The policy in code decides; a case goes through on its own only if the policy approves, the AI committee agrees and the audit passes. Everything else pauses the graph (LangGraph `interrupt`) until an analyst signs or returns it with a reason from a closed list; the sign-off is stored in SQLite and can happen later, from another process. In the backtest 15.1% of applications would go to an analyst: 126–202 analyst hours per 10,000 applications.
- **Fairness:** the public data has no sex, age or race, so direct discrimination cannot be measured. Proxy checks at a 20% rejection cut, measured on Jul–Dec 2015 only, show calibration within ±0.7 points for every housing and region group, even rejection across regions (19.6%–20.2%), and disparate impact by housing (29% of renters rejected vs 11% of mortgage holders), in line with their real default (17.4% vs 12.0%). Credit history length, an age proxy, is never given as a reason; dropping it costs −0.0032 AUC and a monotone constraint −0.0011.
- **Data:** only loans Lending Club approved (2012–2015); no rejected applications or offer acceptance data.
- **Dummy costs:** the P&L costs are illustrative values, not those of a real lender; there are no real funding, operating or capital costs, nor counteroffer acceptance data; the net P&L uses explicit assumptions with a sensitivity table.
- **Small evals:** 35 cases catch large errors, not fine differences between versions.

## What I learned

1. **Value is measured on the decision actually applied, and net.** My first number (+USD 337k) came from a cut the system did not apply. Measured on the real policy and with costs, the policy does not beat a simple subgrade cut, and I would rather show it.
2. **In agents, design matters more than the model.** With the same Haiku, giving everyone the same policy and the same data took agreement from 0% to 100%.
3. **In credit, the LLM should not decide.** Code decides because it is explainable and repeatable, the AI analyzes and writes, and a person signs off the gray zone.
4. **A well-documented negative result is still a result.** The causal effect of the rate could not be identified. I reported it and covered it with a stress scenario instead of inventing a number.
5. **An external review finds what you cannot see.** The review by Sergio left 31 issues, several of them critical, and pushed me to measure better: evals with fixed references, in CI and with a random floor.

## Stack

Python 3.12+ · pandas · scikit-learn · LightGBM · XGBoost (CUDA) · PyTorch · Optuna · TreeSHAP (LightGBM pred_contrib) · survival analysis · two-way fixed effects · double machine learning · LangGraph · Claude (Anthropic SDK) · MCP · SQLite · LangSmith · pytest · GitHub Actions.

How to reproduce, the full pipeline and the agent commands are in the [Spanish README](README.md#cómo-reproducirlo).
