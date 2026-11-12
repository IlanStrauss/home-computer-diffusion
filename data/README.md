# Data

This folder holds the source data in `raw/` and the datasets that the build scripts in `../code/` make from them in `processed/`. Every processed file can be rebuilt from `raw/`; the top-level README gives the run order.

## Getting the raw files

- **Prices** (`raw/prices/`) are in the repository: BEA's NIPA flat file `NipaDataA.txt` (release of 4 November 2026) and the BLS CPI for personal computers, `CUUR0000SEEE01.csv`, from FRED (downloaded 9 November 2026). BEA replaces its flat file with every release, so later downloads will differ.
- **CPS** (`raw/cps/`, 95 MB): run `sh code/get_raw_data.sh` from the top folder. It downloads the eight supplements and the five NBER record layouts that `build_cps.py` reads, then checks them against `raw/CHECKSUMS.sha256`.
- **CE** (`raw/ce/`, 1.3 GB): BLS refuses scripted downloads, so download the 20 comma-delimited Interview Survey files `intrvw94.zip` to `intrvw13.zip` in a browser from https://www.bls.gov/cex/pumd_data.htm into `raw/ce/`, then run `sh code/get_raw_data.sh` again to check them. BLS occasionally reissues these files, and a reissued file will fail the check and may change the results slightly.

## Processed files

| File | One row per | Rows | Built by |
|---|---|---|---|
| `households_combined.csv.gz` | household, both surveys | 604,518 | `build_combined.py` |
| `ce_units.csv.gz` | CE consumer unit, the analysis sample, with its 44 replicate weights | 178,891 | `build_ce_units.py` |
| `ce_households.csv.gz` | CE interview | 554,076 | `build_ce.py` |
| `ce_computer_purchases.csv.gz` | CE computer purchase | 34,293 | `build_ce.py` |
| `cps_households.csv.gz` | CPS household | 425,627 | `build_cps.py` |
| `prices_annual.csv` | year, 1959 to 2025 | 67 | `build_prices.py` |
| `prices_monthly.csv` | month, 1983 to 2015 | 396 | `build_prices.py` |

`build_ce.py` also writes `ce_replicate_weights.csv.gz` (the replicate weights of every CE interview, 66 MB), which `build_ce_units.py` reads. It is too large to keep in the repository and is rebuilt from `raw/ce/`.

## The combined household file

`households_combined.csv.gz` holds the 178,891 CE households interviewed from January 1994 to March 2013 and the 425,627 CPS households interviewed in eight supplements from October 1984 to October 2003, one row per household, with the variables the two surveys share under common names and codes. It is for readers who want the data without the model; the paper's code reads `ce_units.csv.gz` and `cps_households.csv.gz`. Variables that only one survey records are blank for the other survey. Use `weight` for weighted shares and means within a survey and year.

| Variable | Meaning |
|---|---|
| `survey` | `CE` (Consumer Expenditure Survey) or `CPS` (Current Population Survey) |
| `id` | Household identifier: the CE consumer unit number, or the CPS year joined to the file's household identifier |
| `year`, `month` | Interview year and month: the household's first public interview with the appliance inventory (CE) or the supplement month (CPS) |
| `ref_year`, `ref_month` | Year and month that the computer answer refers to: the inventory month (CE, see the reference-month rules below) or the supplement month (CPS) |
| `owns_computer` | 1 if the household had a computer, 0 if not, blank if the household did not answer (0.8% of CPS households) |
| `weight` | Survey weight, scaled to average 1 within each survey and year: the CE's `FINLWT21` with each quarter given an equal share of its year, or the CPS household weight |
| `in_paper` | 1 for the 548,284 households that the paper's estimates use: every CE household, and the 369,393 CPS households that answered the computer question and have a positive weight, an income bracket, a state, and known household traits |
| `age` | Age of the household head in years (the CE's reference person, the person who owns or rents the home, or the CPS reference person) |
| `sex` | Sex of the household head: 1 male, 2 female |
| `race3` | Race of the household head: `white`, `black`, or `other` (including more than one race) |
| `hisp` | 1 if the household head is Hispanic |
| `married` | 1 if the household head is married (CE) or married with spouse present (CPS) |
| `kids` | Number of household members under 18 |
| `persons` | Number of household members (in the CE, members of the consumer unit) |
| `educ4` | Education of the household head: `lt_hs` (less than high school), `hs` (high school), `some_col` (some college, including associate degrees), or `col` (college degree or more) |
| `state` | State FIPS code, blank for the 14.7% of CE households whose state BLS suppresses |
| `region` | Census region: 1 Northeast, 2 Midwest, 3 South, 4 West |
| `spending` | CE only. Long-run spending in current dollars a year: four times the household's mean spending per three-month interview period over all its interviews, leaving out vehicle purchases, computer and internet spending, personal insurance and pensions, and cash contributions |
| `income` | CE only. Family income before taxes in current dollars over the last 12 months, for complete income reporters (`FINCBTAX`, or `FINCBTXM` in 2004 and 2005), blank for the others |
| `income_lo`, `income_hi` | CPS only. Limits of the household's family income bracket in current dollars; `income_hi` is `inf` for the open top bracket |
| `pc_price` | BEA PCE price index for personal computers and peripheral equipment (2017 = 100) in the month `ref_year`, `ref_month`, interpolated in logs between annual values |

## Notes on each source

The starting point for the CE computer item is Schmitt and Wadsworth (2002, CEP Discussion Paper 526): holding education, age, family type, and race fixed, doubling income raised the chance of owning a PC by 1.4 points in 1988 and 3.5 points in 1998 (their Table 6, 0.020 and 0.051 times ln 2). All sources are official public microdata. Raw files are kept unchanged in `raw/`, the code is in `../code/`, and the run logs are in `../output/logs/`.

### Consumer Expenditure Survey (CE), Interview Survey, 1994–2013

- Source: BLS public-use microdata, the comma-delimited yearly files `intrvw94.zip` to `intrvw13.zip` from https://www.bls.gov/cex/pumd_data.htm, kept unchanged in `data/raw/ce/` with the BLS dictionary (`ce-pumd-interview-diary-dictionary.xlsx`). BLS refuses scripted downloads (HTTP 403), so the files are downloaded in a browser. Every zip passed an integrity test. The code is `code/build_ce.py`, the log `output/logs/build_ce.txt`, and the outputs `data/processed/ce_households.csv.gz` (one row per interview), `data/processed/ce_replicate_weights.csv.gz`, and `data/processed/ce_computer_purchases.csv.gz`. `code/build_ce_units.py` turns them into the analysis sample, `data/processed/ce_units.csv.gz` (log `output/logs/build_ce_units.txt`).

- Unit: one row per consumer-unit interview (`NEWID`). The last digit is the interview number (2–5) and the rest identifies the consumer unit. A unit is interviewed up to four times, a quarter apart. From 1994 to 2013, 522,833 interviews with 178,920 consumer units answered the appliance inventory.

- File structure. The household files (FMLI) come by interview quarter, and the appliance files (APL, APB) by yearly release. Each yearly zip runs from the first quarter of its year to the first quarter of the next, so a first quarter is released twice: as `fmliYY1` at the end of one zip and as `fmliYY1x` at the start of the next. The build keeps the copy with the appliance inventory and, of two such copies, the later (x) release; 117,716 duplicate interviews are dropped. The 1994 zip has no 1995 quarter, and the 1995 zip carries 1995Q1 once.

- Ownership: the major-appliance inventory (APL file), code 13. The item is "Computers, not solely for games" from 1994Q1 to 2001Q1 and "Home Computer" from 2001Q2 to 2013Q1 (BLS PUMD dictionary). The last inventory, 2013Q1, is in `intrvw12.zip`; `intrvw13.zip` has no APL file, so 2013 ownership rests on the first quarter only. The public files do not carry the inventory before 1994. Schmitt and Wadsworth used the same question from 1987. ICPSR's archived CE releases list appliance holdings for earlier years (for 1988, study 9451); whether they include computers is unconfirmed, and access needs an ICPSR member institution.

- Coverage of the inventory: 97.4–99.9% of interviews have APL records in each year from 1994 to 2012.

- The inventory is not updated within a unit's panel. Across consecutive interviews of the same unit, ownership never changes, except in 1995 (42 changes) and 1996 (524 changes in 12,654 pairs, about as many each way). The BLS Handbook of Methods (chapter 16) says the first interview, which is not in the public files, collects the inventory of major durables. Ownership may therefore refer to that interview, about one quarter before a unit's first public interview.

- Analysis sample (`ce_units.csv.gz`): one record per unit, at its first public interview with the inventory, with long-run spending from all its interviews. Of 178,920 units, 29 have no positive long-run spending (defined below) and are dropped, leaving 178,891: 6,153 to 13,743 a year, and 2,203 in 2013 (first quarter only). On this basis, ownership in the CPS years is 35.2% in 1997 (CPS 36.8%), 40.8% in 1998 (42.3%), 50.1% in 2000 (51.1%), 56.9% in 2001 (56.6%), and 64.2% in 2003 (61.8%).

- Weights: `FINLWT21` is the population weight of each interview. The household file also carries 44 replicate weights (`WTREP01`–`WTREP44`) for standard errors by balanced repeated replication. For data before 2025, BLS computes the variance as the sum of squared differences between the 44 replicate estimates and the full-sample estimate, divided by 44 (BLS page "CE standard errors"). The factor of four on that page applies only to Fay's method, which BLS uses from 2025 data. In the files checked (1994Q2, 2012Q2), exactly half the replicate-weight cells are blank and the rest are about twice `FINLWT21`, as standard BRR implies; BLS says to read blanks as zero. The build writes them, with blanks set to zero, to `ce_replicate_weights.csv.gz`; in every year exactly half are zero.

- Weights in the analysis sample: `FINLWT21` of the unit's first record, rescaled so each quarter carries an equal share of its year's weight, then scaled to average 1 within each year. Unrescaled, 1994's first quarter would carry 51% of the year's weight, against about 25% in other years, because the public files begin in 1994Q1, so every unit then in the panel (interviews 2–5) is first seen in that quarter: 5,190 of 1994's 9,966 units. The computer item itself is older (Schmitt and Wadsworth use it from 1987). The 44 replicate weights are treated the same way.

- Reference month, when the inventory was taken:
  - for the 131,758 units first seen at interview 2, three months earlier (the first, bounding, interview is not published);
  - for the 42,221 units first seen at interviews 3–5 after 1994, 3 × (interview − 1) months earlier, which assumes a bounding interview followed by missed interviews;
  - for the 4,912 units first seen at interviews 3–5 in 1994 (3,903 of them in 1994Q1, when the public files begin), the month of their first record.
  - The last two rules are assumptions. A household that moves into a sampled address mid-panel may answer the inventory at its first interview, and the item may already have been asked in 1993 bounding interviews, since Schmitt and Wadsworth used it from 1987.

- Weighted ownership by interview year, with our CPS figure in brackets where a CPS survey exists: 1994 23.8%; 1995 26.2%; 1996 30.3%; 1997 34.6% (36.8%); 1998 39.5% (42.3%); 1999 43.6%; 2000 49.2% (51.1%); 2001 55.5% (56.6%); 2002 59.8%; 2003 63.6% (61.8%); 2004 67.7%; 2005 68.3%; 2006 71.4%; 2007 74.0%; 2008 75.8%; 2009 77.4%; 2010 79.1%; 2011 80.5%; 2012 81.0%; 2013 81.0% (first quarter). The CE figure averages interviews spread over the year, while each CPS survey is one month late in the year, so in the fast-rising years the CE runs 1–3 points lower. In 2003 it is 1.8 points higher.

- Income: `FINCBTAX`, family income before taxes over the last 12 months as collected. Only complete income reporters (`RESPSTAT` = 1, 75–87% by year) have full income. BLS dropped `FINCBTAX` and its rank `INC_RANK` from 2004Q2 to 2005, and the 2004 release (which supplies 2004Q1 here) lacks them, so in this file both are missing for all of 2004–2005. `FINCBTXM`, income including BLS's imputation, and its rank `INC_RNKM` start in 2004.

- Spending: total spending in the three-month reference period (`TOTEXPPQ` + `TOTEXPCQ`), food, vehicle outlays (new and used cars and trucks, other vehicles), and computer-related spending from the monthly file. The codes come from the UCC lists in the zips (`ucciYY.txt`):
  - hardware: UCC 690111, plus 690310, installation for computers;
  - software and accessories: 690112, split from 2011Q2 into 690119 software and 690120 accessories;
  - internet services: 690114, computer information services, which first appears in 1996, and 690116, internet services away from home.
  - personal insurance and pensions (`PERINSPQ` + `PERINSCQ`) and cash contributions (`CASHCOPQ` + `CASHCOCQ`). Both have breaks. Cash contributions are 0.8% of total spending in 1994–2000, 2.6% in 2001, and 3.3–3.9% from 2002. Personal insurance and pensions are about 10% in 1994–2003 and 11.3–11.9% from 2004, the year BLS's income imputation starts. We have not checked the causes of either break.

- Long-run spending (`s` in the analysis sample): four times the mean, over all the unit's interviews, of total spending less vehicle purchases, computer and internet spending, personal insurance and pensions, and cash contributions. Taking out the last two also takes out their breaks. The weighted median is $19,220 in 1994 and $31,061 in 2013. Across a unit's interviews, the standard deviation of log spending has a median of 0.19.

- Price paid: the minor-appliance file records each purchase of "Computers, computer systems and related hardware for non-business use" (APB, code 640) with its price (`MINPURX`). Purchases for someone outside the unit (`GFTCMIN` = 3) are set aside.
  - The code includes peripherals such as printers, and the public file cannot separate them. From 2011Q2, computer accessories have their own code (645). Purchases bundled with other items (codes 800 and 900) cannot be assigned to computers.
  - The 2000 release has no APB file, so 2000 prices come only from 2000Q1, which is in the 1999 release. The Stata version of the 2000 release may contain it; this is unchecked.
  - A first-quarter interview's purchases appear in two yearly releases. The build keeps the later release and drops 8,563 repeats. 192 identical rows within a release are kept as separate purchases.
  - Weighted by `FINLWT21`, the median price paid for own use is $400 in 1994, $464 in 1995, $389 in 1996, $214–349 in 1997–2009, and $300–413 in 2010–2013. The 90th percentile falls from $2,150–2,668 in 1994–1996 to $1,100–1,174 in 2011–2013. Code 640 drops accessories from 2011Q2, which may explain the higher 2012 median.

### Computer prices

- Source files, kept unchanged in `data/raw/prices/`:
  - `NipaDataA.txt`: BEA's NIPA flat file of annual values from https://apps.bea.gov/national/Release/TXT/, release of 4 November 2026, downloaded 9 November 2026. `SeriesRegister.txt` at the same address lists the series codes.
  - `CUUR0000SEEE01.csv`: the BLS CPI-U for personal computers and peripheral equipment, U.S. city average, not seasonally adjusted, monthly from December 1997, downloaded from FRED (https://fred.stlouisfed.org/series/CUUR0000SEEE01) on 9 November 2026.

- Series. The code is `code/build_prices.py`, the log `output/logs/build_prices.txt`, and the outputs `data/processed/prices_annual.csv` and `data/processed/prices_monthly.csv`.
  - Main: the BEA PCE price index for personal computers/tablets and peripheral equipment, `DCPPRG` (NIPA Table 2.4.4U, line 49), annual 1977–2025; the 1977 value repeats 1978's. FRED does not carry it under the usual code.
  - Checks: the BEA PCE price index for information processing equipment, `DIPERG` (Table 2.4.4U, line 48), which adds software and other information equipment; BEA's price index for private investment in computers and peripheral equipment, `B935RG` (Table 5.3.4, line 11); the CPI-U above; and the CE price paid.
  - Deflator: the BEA PCE price index for all items, `DPCERG` (Table 2.3.4, line 1), which the CPS extension and the constant-dollar checks use to move dollars between years.

- The main index falls 0.168 log points a year in 1984–93, 0.303 in 1993–2003, and 0.124 in 2003–13, or 4.13 log points in all from 1994 to 2013. The CPI-U falls at the same rate as the main index in 1998–2013 (0.182 a year). The price paid barely falls from 1994 to 2013: the median by 0.18 log points and the 90th percentile by 0.67.

- Monthly values place each annual value at mid-year and interpolate linearly in logs between them. Within each year from 1994 to 2013, the monthly values average to the annual log value within 0.017. The price paid starts in 1994, so the 1,287 units with reference months in late 1993 get its 1994 value.

- Education of the reference person is coded 1–7 up to 1995 and 0/10–17 from 1996. Both are mapped to four groups: less than high school, high school, some college (including associate degrees), and college degree or more. Every interview maps to a group.

- Race and Hispanic origin (BLS dictionary):
  - `race3` is white, black, or other, from `REF_RACE` (1 white, 2 black, 3–6 other groups). A reference person reporting more than one race (code 6, from 2003Q2) is "other", as in the CPS file.
  - `hisp` is 1 if `ORIGIN1` = 2 ("Spanish") up to 2003Q1. From 2003Q2 it is 1 if `HORREF1` gives a Hispanic group (codes 1–6); `HORREF1` is blank for everyone else.
  - `hisp` agrees with `HISP_REF`, which starts in 2009Q2, in 98.7% of 143,032 interviews.
  - The weighted Hispanic share rises from 7.5% (1994) to 9.3% (2002). It dips to 7.8–8.3% in 2003–2005, after the change of question, and reaches 12.6% by 2013.

- `married`: `MARITAL1` = 1. The state is suppressed for 10–18% of interviews.

- Ownership by income fifth (`INC_RANK`, complete reporters; not available in 2004–2005), poorest to richest fifth: 10.5%, 11.1%, 18.6%, 31.1%, and 49.8% in 1994; 37.4%, 46.5%, 64.1%, 78.1%, and 91.2% in 2003; and 57.9%, 73.4%, 87.0%, 93.1%, and 96.3% in 2013.

- Ownership by education, in the order less than high school, high school, some college, and college degree: 5.6%, 15.7%, 30.1%, and 43.7% in 1994; 27.0%, 52.7%, 73.2%, and 85.5% in 2003; and 50.9%, 71.7%, 87.5%, and 93.4% in 2013.

### Current Population Survey (CPS), Computer Use supplements, 1984–2003

- Files for October 1984, 1989, 1993, and 1997 and December 1998 come from NBER (data.nber.org/cps/). Files for August 2000, September 2001, and October 2003 come from Census (www2.census.gov/programs-surveys/cps/datasets/). All are kept in `data/raw/cps/` with their documentation. The code is `code/build_cps.py`, the log `output/logs/build_cps.txt`, and the output `data/processed/cps_households.csv.gz`, one row per household.

- Record layouts for 1997–2003: NBER Stata dictionaries (`*.dct`), checked against the Census documentation.

- Record layouts for 1984, 1989, and 1993: typed from the scanned Census documentation (`cpsoct84.pdf`, `cpsoct89.pdf`, `cpsoct93.pdf`) and checked against the record counts the documentation gives. The pdf pages are listed in the code. The 1984 documentation gives positions as word and character, six characters to a word.

- Household question: "Is there a computer in this household?" (1984, 1989, 1993, 1997), "Is there a personal computer or laptop in this household?" (1998, 2000), "Is there a computer or laptop in this household?" (2001, 2003).

- One row per interviewed household. Person characteristics are those of the reference person (relationship code 1 or 2). In 1984–1993 the files have one record per person with record types: interviewed adult, noninterview, Armed Forces, and child. The reference person is taken from an interviewed-adult or Armed Forces record. The computer answer is a household item repeated on every adult's record, and no household has one adult answering yes and another no. In 1984, 2.5% of households are blank on every adult record. In 1989, 3.2% of households have no answer, and in about 1,400 more some adults are coded "no response" while the others answer no. A household owns a computer if any adult answers yes.

- About 0.3% of interviewed households in 1984–1993 have no record with relationship code 1 or 2 and are dropped (164 in 1984, 169 in 1989, 151 in 1993).

- Weights. From 1989 the files carry a household weight, which is the final weight of the wife in married-couple households and of the reference person otherwise. The 1984 file has no household weight. Its documentation (pdf p. 22) says to use the weight of the principal person, which is defined the same way, so that weight is used. The 1984 supplement weight gives the same ownership rate (8.22% against 8.23%).

- 1984 person characteristics come from two edits on the file. Relationship to the reference person, age, marital status, and sex come from the "new demographic edit" (words 65–66), which is the only edit filled on Armed Forces records. Race, education, and Hispanic origin come from the basic edit on adult records (the new edit leaves them blank there) and from the new edit on Armed Forces records. Age agrees between the two edits for every adult.

- Check against Census report P23-208 (Figure 1), weighted with the household weight. Each line gives our figure, the Census figure in brackets, the share of households answering the question, and the weighted number of households:
  - 1984: 8.2% (8.2%); 97.5% answered; 87.1 million households
  - 1989: 15.0% (15.0%); 96.8%; 93.7 million
  - 1993: 23.0% (22.8%); 100%; 98.5 million
  - 1997: 36.8% (36.6%); 100%; 102.8 million
  - 1998: 42.3% (42.1%); 100%; 104.2 million
  - 2000: 51.1% (51.0%); 100%; 105.7 million
  - 2001: 56.6% (56.3%); 100%; 107.2 million
  - 2003: 61.8% (61.8%); 100%; 113.1 million

- Family income is reported in brackets, and the bracket limits change over time:
  - 1984: 14 brackets, with $15,000–17,499 and $17,500–19,999 split and a top bracket of $75,000 or more.
  - 1989–2001: 14 brackets with a top bracket of $75,000 or more.
  - 2003: 16 brackets with a top bracket of $150,000 or more.
  - `inc_lo` and `inc_hi` give each household's bracket limits in current dollars. The top bracket is open (`inc_hi` is infinite). It holds 2.3% of households in 1984, 21.1% in 2001, and 4.5% in 2003.
  - The share of households with no income bracket rises from 6–8% in 1984–1993 to 19% in 2003.

- Codes made common across years:
  - `educ4`: less than high school, high school, some college (including associate degrees), or college degree or more. 1984 and 1989 ask the highest grade attended and whether it was completed, so years completed are mapped to the four groups (under 12, 12, 13–15, 16 or more). From 1993 the question asks for the highest degree. The change breaks the series between 1989 and 1993, mainly between high school and some college (Jaeger 1997, *Journal of Business & Economic Statistics* 15(3)).
  - `race3`: white, black, or other. In 2003 a person reporting more than one race is "other".
  - `hisp`: Hispanic origin (1/0).
  - `married`: married with spouse present.
  - `kids`: number of household members under 18.
  - `state`: FIPS code. The 1984 file uses the 1960 Census state code, which is mapped to FIPS. The 1989 file carries both codes, and they agree with the mapping.
  - The original codes are kept beside the common ones.

- Standard errors. The public files of these years carry no replicate weights or design codes. The CPS extension takes the CPS sampling error as the probit model's sandwich standard error times √2, an assumed design effect of 2.

- The CPS extension imputes spending from the income bracket. In the five years both surveys cover, thresholds built on imputed spending fall more slowly than measured ones, by about a third when the imputation is applied to CE units themselves. CPS and CE thresholds are therefore not on the same footing (Appendix C.6 of the paper).
