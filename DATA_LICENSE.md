# Dataset provenance and licensing

This reproducibility package uses the DBLP-Scholar and Abt-Buy entity-
resolution benchmark datasets distributed by the Database Group Leipzig at
Leipzig University.

The benchmark source states that these datasets are made available under a
Creative Commons license and requests attribution to the benchmark collection
and its associated publication.

This repository does not assign a new license to these third-party benchmark
data. Users should consult the original dataset source for the applicable
reuse and redistribution terms.

The datasets are used here solely as the benchmark inputs required to
reproduce the experiments reported in the accompanying manuscript.

## Benchmark reference

Hanna Köpcke, Andreas Thor, and Erhard Rahm.
Evaluation of Entity Resolution Approaches on Real-World Match Problems.
Proceedings of the VLDB Endowment, 2010.

## Files used in this package

DBLP-Scholar:

- `data/dblp/DBLP1.csv`
- `data/dblp/Scholar.csv`
- `data/dblp/DBLP-Scholar_perfectMapping.csv`

Abt-Buy:

- `data/abt_buy/Abt.csv`
- `data/abt_buy/Buy.csv`
- `data/abt_buy/abt_buy_perfectMapping.csv`

The scientific code, documentation and other material authored for this
reproducibility package are separate from the licensing of these third-party
benchmark datasets.

## Original licensing statement

The official Database Group Leipzig benchmark page states that the binary
entity-resolution datasets are made available "under the Creative Commons
license." The page does not identify a specific Creative Commons license
variant. This package therefore does not infer or assign a more specific
Creative Commons license to those datasets.

Original source:
https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution

Source checked: 25 September 2026.
