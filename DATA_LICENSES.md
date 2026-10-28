# Data attribution and licensing

The source data are distributed by the UCI Machine Learning Repository under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). Source archives are preserved byte for byte and their SHA-256 hashes appear in `data/raw/manifest.json`. Retrieval date: 27 October 2026.

- V. Lohweg. Banknote authentication. UCI Machine Learning Repository, 2012. [Dataset DOI](https://doi.org/10.24432/C55P57).
- M. Hopkins, E. Reeber, G. Forman, and J. Suermondt. Spambase. UCI Machine Learning Repository, 1999. [Dataset DOI](https://doi.org/10.24432/C53G6X).
- R. Bock. MAGIC gamma telescope. UCI Machine Learning Repository, 2004. [Dataset DOI](https://doi.org/10.24432/C52C8B).
- E. Alpaydin and F. Alimoglu. Pen-based recognition of handwritten digits. UCI Machine Learning Repository, 1996. [Dataset DOI](https://doi.org/10.24432/C5MG6K).
- D. Slate. Letter recognition. UCI Machine Learning Repository, 1991. [Dataset DOI](https://doi.org/10.24432/C5ZP40).
- P. Cortez, A. Cerdeira, F. Almeida, T. Matos, and J. Reis. Wine quality. UCI Machine Learning Repository, 2009. [Dataset DOI](https://doi.org/10.24432/C56S3T).

Processed data remove exact feature-label duplicate rows, recode class labels, and assign identical features a common split group. Wine adds a binary red/white indicator and concatenates both source files. Pendigits concatenates its two tabular split files. Predictions, split identifiers, and audit outputs derive from these attributed datasets. The processed and derived data are redistributed under CC BY 4.0. The original authors and repository do not endorse this study.
