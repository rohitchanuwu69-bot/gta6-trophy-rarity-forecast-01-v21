# Trophy Dilution Forecaster v21

[![make.fun bounty](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![entry rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rohitchanuwu69-bot/gta6-trophy-rarity-forecast-01-v21/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![entry reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rohitchanuwu69-bot/gta6-trophy-rarity-forecast-01-v21/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Entry for the [make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01): forecast how many GTA VI players earn each trophy in week one.

## What it does

`main.py` reads a case (every Exophase read the house recorded before the
checkpoint) and writes one forecast per trophy: five quantiles of its earn
rate at the case target, printed to 2 decimals.

It works in two regimes. When the history is flat, it models the next
refresh from the one refresh the rules document (638 to 3868 players
tracked): earned counts stay, two players join the first 29 trophies in
list order, and the trophies with the top three distinct sub-100 rates
gain a share of the newcomers. When the history moves, it finds the last
refresh jump, fits log players and log earned counts against log time
since release midnight on the reads after it, and reads the fits at the
target; if the last three reads are flat, the median is their level.
The spread is a small floor in the flat regime, and floor plus horizon
growth plus fit residual in the moving regime.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=output python3 main.py
```

Python 3.13, standard library only, no network. Reads every `*.in` in
the input directory and writes one `*.out` per case to the output
directory, deterministic and well inside the 8-second limit per case.
