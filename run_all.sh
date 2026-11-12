#!/bin/sh
# Run every script in order, from the raw files to the figures and tables of the paper.
#   sh run_all.sh            builds the datasets from data/raw/ (see data/README.md), then runs the analysis
#   sh run_all.sh analysis   starts from the datasets in data/processed/ and runs the analysis only
# Each script writes its log to output/logs/. About 75 minutes on a 10-core laptop, an hour of it for the joint model.
set -e
cd "$(dirname "$0")"
mkdir -p output/logs output/results output/diagnostics writeup/figures writeup/tables

run() {  # run code/<script> [arguments], with its log in output/logs/<script>.txt
  s=$1
  shift
  echo "$(date +%H:%M:%S) $s $*"
  case "$s" in
    *.R) Rscript "code/$s" "$@" > "output/logs/${s%.*}.txt" ;;
    *) python3 -u "code/$s" "$@" > "output/logs/${s%.*}.txt" ;;
  esac
}

if [ "$1" != "analysis" ]; then
  run build_cps.py
  run build_ce.py
  run build_prices.py
  run build_ce_units.py
  run build_combined.py
fi
run explore_data.py
run link_choice.py
run first_stage.py
run second_stage.R
run second_stage_constant.R
run income_price.py
run education_raw_gap.py
run cps_extension.py
# The two dates are when CPS computer ownership passed 20% and 30%, as cps_extension.py prints them.
run cps_switch.R 1992.44 1995.96
run joint_prepare.py
run joint_model.py
run joint_design_effect.py
python3 code/figures.py
python3 code/tables.py
echo "$(date +%H:%M:%S) Done. To compile the paper and the online appendix, see step 4 of README.md."
