#!/bin/sh
# Download the raw CPS files and check every raw file against data/raw/CHECKSUMS.sha256.
# Run from the top folder: sh code/get_raw_data.sh
# The CE files must be downloaded in a browser first (BLS refuses scripted downloads): the comma-delimited
# Interview Survey files intrvw94.zip to intrvw13.zip from https://www.bls.gov/cex/pumd_data.htm, saved in data/raw/ce/.
set -e
cd "$(dirname "$0")/../data/raw"
mkdir -p ce cps

NBER=https://data.nber.org
CENSUS=https://www2.census.gov/programs-surveys/cps/datasets
for url in \
    $NBER/cps/cpsoct84.zip $NBER/cps/cpsoct89.zip $NBER/cps/cpsoct93.zip $NBER/cps/cpsoct97.zip $NBER/cps/cpsdec98.zip \
    $CENSUS/2000/supp/aug00supp.zip $CENSUS/2001/supp/sep01supp.zip $CENSUS/2003/supp/oct03supp.zip \
    $NBER/progs/cps/cpsoct97.dct $NBER/progs/cps/cpsdec1998.dct $NBER/progs/cps/cpsaug00.dct \
    $NBER/progs/cps/cpssep01.dct $NBER/progs/cps/cpsoct03.dct; do
  f=cps/$(basename "$url")
  if [ ! -s "$f" ]; then
    echo "Downloading $url"
    curl -fsSL -o "$f" "$url"
  fi
done

missing=0
bad=0
while read -r sum f; do
  if [ ! -f "$f" ]; then
    echo "MISSING  $f"
    missing=$((missing + 1))
  elif [ "$(shasum -a 256 "$f" | cut -d' ' -f1)" != "$sum" ]; then
    echo "CHANGED  $f (differs from the file the paper used)"
    bad=$((bad + 1))
  fi
done < CHECKSUMS.sha256
echo "$(wc -l < CHECKSUMS.sha256 | tr -d ' ') raw files listed: $missing missing, $bad changed."
if [ "$missing" -gt 0 ]; then
  echo "Download the missing CE files in a browser from https://www.bls.gov/cex/pumd_data.htm into data/raw/ce/."
fi
