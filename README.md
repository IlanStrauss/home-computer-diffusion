# How Will Paid AI Spread? Evidence from Home Computers in US Households, 1984–2013

This repository holds the paper, its data, and the code that builds every figure and table in it from the original survey files. The paper is [writeup/household_pc_results.pdf](writeup/household_pc_results.pdf) (draft of November 2026).

## Abstract

The spread of home computers through US households is a guide to how paid AI may spread. We study that spread with 178,891 households interviewed in the Consumer Expenditure Survey from January 1994 to March 2013, the Bureau of Economic Analysis computer price index, and, for the years back to 1984, 369,393 households in eight Current Population Survey supplements from October 1984 to October 2003. Our findings cover 1994 to 2012, since 2012 is the last full year of Consumer Expenditure Survey data. Over those years the share of US households that owned a computer rose from 24% to 81%. We find three key results. First, falling quality-adjusted computer prices, far more than rising incomes, paced that rise. Falling prices account for 53 of its 57 percentage points and rising household spending for 3, even though the median price paid for a computer fell only 7% from 1994 to 2012. Second, the gap in computer ownership between the richest and poorest fifths of households, ranked by income, widened from 39 percentage points in 1994 to 54 in 2003, then narrowed to 37 in 2012. Every household faced the same computer price, but a price cut is most likely to tip households close to affording a computer into buying one, and the households close to affording one were richer households in the 1990s and poorer households in the 2000s. Third, falling prices left the education gap in computer ownership in place. The gap between households headed by college and high-school graduates grew from 30 percentage points in 1994 to 39 in 1997 and narrowed only after 2001, once most college households owned a computer, and among households with equal spending it did not narrow from 1994 to 2012. If paid AI follows this pattern, policymakers will need a quality-adjusted price index for AI services to track how affordable AI is becoming. Once most households pay for AI, a given income gain would raise paid AI use most if it went to the bottom half of households, but cheaper AI alone may leave more educated households ahead.

## What is in the repository

- `data/` holds the data, documented in [data/README.md](data/README.md).
  - `raw/` holds the original downloads, kept unchanged. The BEA and BLS price files are in the repository. The survey files are too large for GitHub, and `code/get_raw_data.sh` downloads or checks them.
  - `processed/` holds the datasets that the build scripts make.
- `code/` holds every script, each run from the top folder.
- `output/` holds what the analysis writes:
  - `results/` has the estimates, as CSV files that `figures.py` and `tables.py` read;
  - `logs/` has the log of each script, and every number in the paper appears in one of them;
  - `diagnostics/` has the plots we checked before and after fitting.
- `writeup/` holds the paper, `household_pc_results.tex` and its compiled PDF, with the `figures/` and `tables/` it inputs.
- `run_all.sh` runs every script in order.

## The combined household file

For readers who want the data without the model, `data/processed/households_combined.csv.gz` (15 MB) puts both surveys in one file of 604,518 households, one row each. It has the 178,891 Consumer Expenditure Survey (CE) households interviewed from January 1994 to March 2013 and the 425,627 Current Population Survey (CPS) households interviewed in eight supplements from October 1984 to October 2003. The two surveys share common variable names and codes: computer ownership, survey weight, age, sex, race, Hispanic origin, marriage, children, household size, education, state, and region. The file adds long-run spending and income in dollars for the CE, income brackets for the CPS, and the BEA price index for personal computers in the month each household's computer answer refers to. The codebook is in [data/README.md](data/README.md).

## Data sources

All data come from US government statistical agencies and are public. We had legitimate access to every source, and every source permits redistribution of files built from it.

| Data | Source | In this repository |
|---|---|---|
| Consumer Expenditure Survey, Interview Survey public-use files, 1994–2013 (`intrvw94.zip` to `intrvw13.zip`, comma-delimited) | Bureau of Labor Statistics, https://www.bls.gov/cex/pumd_data.htm | No (1.3 GB). Download in a browser into `data/raw/ce/`; BLS refuses scripted downloads |
| Current Population Survey, Computer Use supplements: October 1984, 1989, 1993, and 1997, and December 1998 | Census Bureau files as copied by NBER, https://data.nber.org/cps/ | No (95 MB with the record layouts). `code/get_raw_data.sh` downloads them |
| Current Population Survey, Computer Use supplements: August 2000, September 2001, and October 2003 | Census Bureau, https://www2.census.gov/programs-surveys/cps/datasets/ | No. `code/get_raw_data.sh` downloads them |
| Record layouts of the 1997–2003 CPS files (Stata dictionaries) | NBER, https://data.nber.org/progs/cps/ | No. `code/get_raw_data.sh` downloads them |
| NIPA flat file of annual values, `NipaDataA.txt`, release of 4 November 2026 | Bureau of Economic Analysis, https://apps.bea.gov/national/Release/TXT/ | Yes, in `data/raw/prices/`, because BEA replaces the file with each release |
| CPI-U for personal computers and peripheral equipment, `CUUR0000SEEE01` | Bureau of Labor Statistics, via FRED, https://fred.stlouisfed.org/series/CUUR0000SEEE01 | Yes, in `data/raw/prices/` |

`data/raw/CHECKSUMS.sha256` lists the SHA-256 checksum of every raw file the code reads, and `code/get_raw_data.sh` reports any file that is missing or differs from the one the paper used. Figure 9 also plots published numbers on generative AI use from Bick, Blandin, and Deming (Federal Reserve Bank of St. Louis, 13 November 2025, and NBER Working Paper 32966), which `code/figures.py` types in with their sources.

## Licenses

The code is under the MIT License ([LICENSE](LICENSE)). The data files in `data/processed/` and `output/results/` are under the Creative Commons Attribution 4.0 International License ([data/LICENSE](data/LICENSE)). The raw files are works of the US government and in the public domain.

## Requirements

- Python 3.9.6 with numpy 1.26.4, pandas 2.3.3, scipy 1.13.1, statsmodels 0.14.6, matplotlib 3.9.4, and cmdstanpy 1.3.0 with CmdStan 2.39.0.
- R 4.5.1 with brms 2.23.0, rstan 2.32.7, loo 2.8.0, and posterior 1.6.1.
- `curl`, `unzip`, and `shasum`, and, for the paper, `tectonic` or another LaTeX distribution with the `caption` package.

To install the Python and R packages:

```sh
python3 -m pip install -r requirements.txt
python3 -c "import cmdstanpy; cmdstanpy.install_cmdstan(version='2.39.0')"
Rscript -e 'install.packages(c("brms", "rstan", "loo", "posterior"))'
```

The random seeds are fixed. The Bayesian fits use seed 20261109 (`second_stage.R`, `second_stage_constant.R`, `cps_switch.R`, and `joint_model.py`), the cross-validation folds of `link_choice.py` use seed 1994, and the fake-data checks of `first_stage.py` use seeds 20261109 and 20261110.

The full run takes about 90 minutes on a laptop with a 10-core Apple M4 processor and 24 GB of memory, 55 of them for the joint model. The raw files take 1.4 GB of disk space, and the processed data and outputs take 165 MB.

## How to reproduce the paper

1. Clone the repository:
   ```sh
   git clone https://github.com/IlanStrauss/home-computer-diffusion.git
   cd home-computer-diffusion
   ```
2. Get the raw files. Download the 20 CE files `intrvw94.zip` to `intrvw13.zip` (Interview Survey, comma-delimited) in a browser from https://www.bls.gov/cex/pumd_data.htm into `data/raw/ce/`. Then run `sh code/get_raw_data.sh`, which downloads the CPS files and checks every raw file against `data/raw/CHECKSUMS.sha256`.
3. Run `sh run_all.sh`. To skip the raw files and start from the datasets in `data/processed/`, run `sh run_all.sh analysis` instead.
4. Compile the paper with `cd writeup && tectonic household_pc_results.tex`.

## Scripts, in run order

| Script | What it does | Writes |
|---|---|---|
| `build_cps.py` | Reads the eight CPS supplements: one row per household, with the computer answer, the income bracket, and the traits of the household head | `data/processed/cps_households.csv.gz` |
| `build_ce.py` | Reads the CE files: one row per interview, the replicate weights, and the computer purchases | `data/processed/ce_households.csv.gz`, `ce_replicate_weights.csv.gz`, `ce_computer_purchases.csv.gz` |
| `build_prices.py` | Builds the annual and monthly computer price series | `data/processed/prices_annual.csv`, `prices_monthly.csv` |
| `build_ce_units.py` | Builds the CE analysis sample, one row per household, with long-run spending and the inventory month | `data/processed/ce_units.csv.gz` |
| `build_combined.py` | Puts both surveys in one household file | `data/processed/households_combined.csv.gz` |
| `explore_data.py` | Plots the data before any fit | `output/diagnostics/explore_data.pdf` |
| `link_choice.py` | Compares link functions by cross-validated log score | `output/results/link_choice.csv`, `link_choice_cv.csv`; `output/diagnostics/link_choice.pdf` |
| `first_stage.py` | Fits the yearly spending thresholds of the baseline regression and its extensions, with replicate standard errors and fake-data checks | `output/results/first_stage.csv`, `first_stage_replicates.csv.gz`, `first_stage_fake.csv` |
| `second_stage.R` | Fits the second stage, the spending thresholds on the computer price and a trend | `output/results/second_stage.csv`, `second_stage_fitted.csv`; `output/diagnostics/second_stage.pdf` |
| `second_stage_constant.R` | Refits the second stage in constant dollars, a check | log only |
| `income_price.py` | Splits the rise in computer ownership between the price and household spending, and tests their interaction | `output/results/income_price_years.csv`, `income_price_percentiles.csv`, `income_price_paths.csv` |
| `education_raw_gap.py` | Computer ownership by education of the household head, with no model | log only |
| `cps_extension.py` | Builds the CPS spending thresholds back to 1984 and checks them against the CE | `output/results/cps_thresholds.csv`, `cps_validation.csv` |
| `cps_switch.R` | Tests for a switch in the trend at 20–30% ownership | `output/results/cps_switch.csv`; `output/diagnostics/cps_switch.pdf` |
| `joint_prepare.py`, `joint_model.py` (with `joint_model.stan`) | Fits both stages as one Bayesian model, a robustness check | `output/results/joint_units.csv.gz`, `joint_years.csv`, `joint_groups.csv`, `joint_model.csv` |
| `joint_design_effect.py` | Compares the joint model's standard errors with the replicate standard errors | log only |
| `figures.py`, `tables.py` | Write the figures and tables of the paper | `writeup/figures/`, `writeup/tables/` |

`hhpc.py` holds the pieces the Python scripts share. In the code, the regressions keep the labels of the analysis plan: `s1` is the baseline regression, `s2` and `s3` add the noise correction and the household traits, and `s4` is the extended regression. Appendix D of the paper maps the plan to the paper.

## Figures and tables in the paper

| In the paper | File | Made by |
|---|---|---|
| Figure 1 | `writeup/figures/ownership.pdf` | `figures.py` |
| Figure 2 | `writeup/figures/thresholds.pdf` | `figures.py` |
| Figure 3 | `writeup/figures/price_effect.pdf` | `figures.py` |
| Figure 4 | `writeup/figures/paths.pdf` | `figures.py` |
| Figure 5 | `writeup/figures/prices.pdf` | `figures.py` |
| Figure 6 | `writeup/figures/fifths.pdf` | `figures.py` |
| Figure 7 | `writeup/figures/income_price.pdf` | `figures.py` |
| Figure 8 | `writeup/figures/education.pdf` | `figures.py` |
| Figure 9 | `writeup/figures/ai_pc.pdf` | `figures.py` |
| Figure 10 | `writeup/figures/b_and_d.pdf` | `figures.py` |
| Figure 11 | `writeup/figures/cps_ce.pdf` | `figures.py` |
| Figure 12 | `writeup/figures/switch.pdf` | `figures.py` |
| Figure 13 | `writeup/figures/joint.pdf` | `figures.py` |
| Tables 1–3 | in `writeup/household_pc_results.tex`, from the logs in `output/logs/` | |
| Table 4 | `writeup/tables/tab_second_stage.tex` | `tables.py` |
| Table 5 | `writeup/tables/tab_education.tex` | `tables.py` |
| Table 6 | `writeup/tables/tab_checks.tex` | `tables.py` |
| Table 7 | `writeup/tables/tab_concepts.tex` | `tables.py` |
| Table 8 | `writeup/tables/tab_switch.tex` | `tables.py` |
| Table 9 | `writeup/tables/tab_joint.tex` | `tables.py` |
| Table 10 | `writeup/tables/tab_links.tex` | `tables.py` |
| Table 11 | `writeup/tables/tab_cps.tex` | `tables.py` |
| Table 12 | `writeup/tables/tab_validation.tex` | `tables.py` |
| Table 13 | `writeup/tables/tab_thresholds.tex` | `tables.py` |

## How to cite

Ilan Strauss (2026), "How Will Paid AI Spread? Evidence from Home Computers in US Households, 1984–2013", working paper, https://github.com/IlanStrauss/home-computer-diffusion.

## Data citations

- US Bureau of Labor Statistics (2026). Consumer Expenditure Surveys, Interview Survey public-use microdata, 1994–2013. https://www.bls.gov/cex/pumd_data.htm (downloaded 9 November 2026).
- US Census Bureau (1984–2003). Current Population Survey, October 1984, October 1989, October 1993, October 1997, December 1998, August 2000, September 2001, and October 2003 supplements. NBER copies at https://data.nber.org/cps/ and Census files at https://www2.census.gov/programs-surveys/cps/datasets/ (downloaded 4 November 2026).
- US Bureau of Economic Analysis (2026). National Income and Product Accounts, flat files, release of 4 November 2026. https://apps.bea.gov/national/Release/TXT/ (downloaded 9 November 2026).
- US Bureau of Labor Statistics (2026). Consumer Price Index for All Urban Consumers: Personal Computers and Peripheral Equipment in U.S. City Average [CUUR0000SEEE01], retrieved from FRED, Federal Reserve Bank of St. Louis. https://fred.stlouisfed.org/series/CUUR0000SEEE01 (downloaded 9 November 2026).

This README follows the [template README for social science replication packages](https://social-science-data-editors.github.io/template_README/) of the Social Science Data Editors.
