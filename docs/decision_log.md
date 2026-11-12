# Decision log

Produced by notebooks/01_explore_data.ipynb. It contains no household data.

1. Population: keep only households with hh9 = Filled (7,407 of 7,824, 94.7%). interview_status is a review workflow, not completion, so it is not used.
2. Dropped 399 'Not home' households: no data, target missing. Non-response is concentrated in BUJUMBURA (48% vs 24%), so the model under-represents these households (limitation).
3. Dropped 18 'Partially filled' households: m11cons is truncated (mean 6.9 vs 13.7), so the poverty label would be unreliable.
4. Train/test split must be grouped by cluster (hh1), because households in a cluster resemble each other.
5. Text columns (identifiers, names, timestamps, free text) are never predictors.
6. Target: poor = 1 if m11cons <= 12 (40th percentile of the kept data). Label is relative to this data, not an official poverty line. Share poor: 43.9%.
7. Wealth index (wlthscor, wlthind5) rejected as target: predictable from the housing and asset variables (test R2 1.00, tree model, cluster-grouped split), so it is circular. m11cons is far less predictable (test R2 0.38). A linear model failed on these columns because category codes and real quantities are mixed; to be fixed in Step 6.
8. Mixed variables (18): 'Not known' codes (98, 998) and values above them become empty; '97 and above' stays as a number; ws3 code 995 (water on premises) becomes 0 minutes; tn5 code 95 (more than 24 months) becomes 25. The 56 numeric variables need no cleaning.
9. Any value above the highest labelled code is invalid and becomes empty in all proxy variables (e.g. hc9_6 = 120,000, hc9_18 = 30,000,000: data-entry errors). The Step 6, Part 2 test (5% unlabelled) was too lenient to catch them.
10. Skipped counts: livestock (hc14_1 to hc14_8), land area (hc12) and nets (tn2, tn3) are set to 0 when the gate answer is No (hc13, hc11, tn1).
11. Other structural gaps (water treatment ws6_*, net details tn3a_*, tn4, tn5, tn6, chimney hc7a) stay empty: a skipped question is not a zero.
12. Real gaps ('Not known' codes) stay empty; imputation is learned on the training data only, inside the modelling pipeline, to avoid leakage.
13. tn8_0 to tn8_15 (16 columns) are unlabelled and look like a list of household member numbers: excluded in Step 7.
14. Final predictors: 111 variables (housing and dwelling ownership, water and sanitation, nets, assets, head of household age, sex, marital status and education, household size and composition, productive assets, location).
15. Identity variables (religion hc1a, religious practice hc1aa, mother tongue hc1b) are excluded: a targeting tool should not rely on identity, for fairness and data protection. Sex of the head (hhsex) is kept, and fairness by sex is checked at evaluation.
16. Label check (Step 8): the poverty rate is 46.7% in rural and 17.1% in urban households. It falls with the head's education (53.9% none, 37.9% primary, 23.1% secondary) and with owning a phone (55.8% vs 28.9%), a radio (48.4% vs 28.3%) or a bicycle (46.3% vs 24.4%). The label behaves as expected.
17. Caveat (Step 8): the poverty rate is highest in the smallest households (48.3% for 1 to 3 members vs 40.4% for 8+), because m11cons counts items per household and not per person. Reported as a limitation of the label.