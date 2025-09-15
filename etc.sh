#!/bin/bash

# Dates you want commits for (2025 instead of 2024)
DATES=("2025-08-11" "2025-08-12" "2025-08-18" "2025-08-31" "2025-09-07")

# Loop through each date
for DATE in "${DATES[@]}"; do
  # Random number of commits (between 4 and 6)
  NUM_COMMITS=$((RANDOM % 3 + 4))

  for ((i=1; i<=NUM_COMMITS; i++)); do
    echo "Commit $i on $DATE" >> commits.txt
    git add commits.txt
    GIT_COMMITTER_DATE="$DATE 12:$i:00" \
    git commit --date "$DATE 12:$i:00" -m "Commit $i on $DATE"
  done
done
