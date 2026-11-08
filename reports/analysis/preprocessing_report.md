# Preprocessing Report

Source file: BotPrint_Research_Dataset.xlsx
Original rows: 1,196
Cleaned rows: 1,196
Excluded rows: 0

## Transformations applied
1. Timestamp converted from epoch milliseconds to datetime (Timestamp_dt).
2. Events sorted chronologically within each SessionID.
3. Behavioural columns coerced to numeric; non-convertible values -> NaN (reported).
4. Missing values handled per event type; structural nulls left as NaN (not zero).
5. Invalid measurements investigated (negative speed, negative elapsed, null coords on mouse).
6. Invalid records excluded and documented with reasons.
7. Labels/metadata kept separate from behavioural measurements (no target leakage).

## Exclusion reasons
- None (no records excluded)

## Output files
- dataset/processed/cleaned_events.csv (full cleaned dataset)
- dataset/processed/cleaned_events_ids.csv
- dataset/processed/cleaned_events_labels.csv
- dataset/processed/cleaned_events_metadata.csv
- dataset/processed/cleaned_events_behavioural.csv
- dataset/processed/excluded_records.csv (if any exclusions)
- reports/analysis/preprocessing_before_after.csv
- reports/analysis/preprocessing_assertions.csv
- reports/analysis/preprocessing_numeric_validation.csv
- reports/analysis/preprocessing_missing_by_event_type.csv
- reports/analysis/preprocessing_invalid_measurements.csv
- reports/analysis/preprocessing_target_leakage_check.csv

## Original data
The original Excel file (dataset/raw/BotPrint_Research_Dataset.xlsx) was not modified.