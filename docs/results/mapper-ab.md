# Mapper A/B on identical trajectories (evaluation only)

`scripts/dev/mapper_ab.py`: the robot explores with the default mapper; a shadow
mapper with one setting changed receives the same keyframes and every other map
input into its own grid. Both maps are scored against ground truth after 240 s.

| room | setting compared | default: false-free (deep) / coverage | alternative: false-free (deep) / coverage |
|---|---|---|---|
| heldout_c | obstacle bases need 2 agreeing keyframe pairs (default) vs 1 | 195 (84) / 65.0 % | 188 (84) / 64.5 % |
| home_a | same | 64 (15) / 38.7 % | 60 (14) / 38.5 % |
| heldout_c | no free evidence above obstacle bases: off (default) vs on | 265 (142) / 70.5 % | 265 (142) / 70.5 % |

Conclusions: requiring two agreeing pairs removes about 80 % of stray obstacle hits
(`scripts/dev/obstacle_hits_probe.py`) but does not measurably change the final map;
suppressing free evidence above detected bases has no effect, so it is off. The
false-free cells inside obstacles come from the parallax floor test itself
(`scripts/dev/false_free_probe.py`: 0.2–0.5 % of floor hits land inside obstacles);
this remains open.
