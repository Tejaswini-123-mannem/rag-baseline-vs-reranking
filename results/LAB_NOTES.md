# Measured lab notes - learning support, not your submission report

Corpus: 118 chunks from one PDF; 160 words per chunk and 40-word overlap.
All 17 queries were measured; paired LLM answers were generated for two recoveries and one regression.
The other 14 queries are retrieval-only diagnostics. No success rate is claimed.

| Query | Baseline median ms | Cross-encoder median ms | Advanced median ms |
|---|---:|---:|---:|
| If I have four progress cards and draw a fifth outside my turn, where must the discarded card go? | 14.00 | 349.42 | 361.99 |
| Can I move the robber by playing Bishop before the first barbarian attack? | 14.62 | 475.03 | 489.75 |
| Can I activate a knight and use it to drive away the robber on the same turn? | 13.20 | 381.66 | 395.53 |
| If I only have a metropolis and settlements, do I lose my metropolis when barbarians win? | 11.70 | 351.00 | 362.70 |
| Can I promote a knight twice during the same turn? | 11.20 | 349.28 | 360.48 |
| Do city walls protect my city from being destroyed by barbarians? | 10.81 | 332.35 | 341.26 |
| Can I keep a fifth progress card if I draw it when it is not my turn? | 12.68 | 349.32 | 362.00 |
| Can I use the Warlord card to activate knights that I built this turn? | 11.74 | 325.81 | 336.26 |
| When the barbarians win, does the player with the weakest army always lose a city? | 13.08 | 346.63 | 361.16 |
| Does a settlement produce commodities when its number is rolled? | 11.06 | 340.25 | 350.66 |
| Can I use a resource harbor to exchange commodities at the two to one rate? | 12.96 | 380.53 | 393.49 |
| If defenders tie for the strongest army, who receives a Defender of Catan card? | 12.98 | 334.66 | 347.72 |
| Can I buy city improvements if I no longer have any cities? | 11.29 | 337.85 | 347.66 |
| Can I move a displaced knight to a new intersection that is not connected to my roads? | 13.14 | 355.37 | 369.48 |
| Does the Merchant Fleet card give me the Merchant victory point? | 10.41 | 381.68 | 393.61 |
| Can a knight displace another knight of equal strength? | 11.24 | 366.51 | 377.75 |
| Can I steal a metropolis by reaching level four after its owner reaches level five? | 14.96 | 527.10 | 542.05 |

Timing: five warm repetitions; excludes model loading, indexing, and generation.
Advanced includes query embedding, vector search, cross-encoder scoring, and sorting.
Medians of component times need not sum to the median of total time.

## Verified evidence recoveries

### If I have four progress cards and draw a fifth outside my turn, where must the discarded card go?
Passage p09_0360: rank 10 -> 1.
Page 9: discard one card of your choice to the bottom of the appropriate stack. The baseline passage p09_0240 ends before the destination. This is a chunk-boundary omission.

**Actual baseline answer:**
[p05_0000; page 5] 55 turn o vervie W The player who placed the last city takes the first turn; afterwards, in clockwise direction, one by one the other players take their turns. On your turn, the following actions are performed, in the order listed: • Y ou may never hold more than four progress cards in your hand. If you draw a fifth progress card, and you cannot play it because it is not your turn

**Actual re-ranked answer:**
To the bottom of the appropriate stack.

Generation time: baseline 13302.05 ms; advanced 2582.78 ms.

Observed answer difference: baseline copies unrelated turn information and does not provide the discard destination. The re-ranked answer gives the destination directly. This is a recovered omission, not evidence that every answer improved.

### Can I buy city improvements if I no longer have any cities?
Passage p12_0000: rank 10 -> 1.
Page 12: existing improvements remain, but further improvements require owning at least one city again. Baseline passages discuss ordinary purchases and metropolis restrictions without this zero-city condition.

**Actual baseline answer:**
You may not purchase any improvements beyond the third level of a given color unless you have a city where you could build a metropolis. If you do not have such a city on the board, you must wait until you have built at least one more city.

**Actual re-ranked answer:**
Once you own a city again, you may continue to purchase city improvements normally.

Generation time: baseline 11312.50 ms; advanced 3338.81 ms.

Observed answer difference: baseline discusses the restriction on improvements above level three, which does not resolve whether lower-level purchases are allowed with zero cities. The re-ranked answer states that purchases resume once a city is owned again. It is terse and omits a citation, but addresses the missing condition.

## Regression to discuss
The activate-and-use-a-knight question promotes p13_0480, an optional pre-roll variant. The ordinary activation restriction is on page 6. This illustrates that re-ranking may prefer superficially relevant exceptions or variants. Read the paired answers below.

**Actual baseline answer:**
Any knight may do so, regardless of its strength. A knight may only chase away a robber if the robber is on one of the three hexes adjacent to that knight. When you chase away the robber, move it to any numbered hex and steal one resource from any player who has a settlement or city adjacent to that hex (as if you had played a knight card from the Cat

**Actual re-ranked answer:**
You may use one of your active knights to chase away the robber. Any knight may do so, regardless of its strength. A knight may only chase away a robber if the robber is on one of the three hexes adjacent to that knight. When you chase away the robber, move it to any numbered hex and steal one resource from any player who has a settlement or city adjacent to that 

## How to explain the mechanism
Bi-encoder: precompute independent document vectors; encode the query and compare vectors. Cross-encoder: jointly process each query and passage, allowing token interactions before scoring.
Dense retrieval reduces the candidate set. Cross-encoding all N chunks needs N joint forward passes per query; this experiment uses only ten pairs. Re-ranking cannot recover evidence outside those ten.

## What you must do before submission
Read the cases and source pages, then write your concise report in your own words using REPORT_WORKSHEET.md. The notebook and data support your explanation. The local files have not been committed or pushed.