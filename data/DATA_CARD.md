# PaySim Data Card

## Status

Phase 1 implementation supports a local PaySim CSV. The raw file is not committed.

## Expected source

Use the publicly available PaySim `PS_20174392719_1491204439457_log.csv` file. Record the exact download URL, SHA-256 checksum, download date, and row count here after obtaining it.

## Sensitive data

PaySim is synthetic. `nameOrig` and `nameDest` are synthetic identifiers, not real customer identities. Do not treat model performance on this data as evidence of production banking performance.

## Required schema

`step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, and `isFlaggedFraud`.

## Split policy

Rows are ordered by `step`: 70% train, 15% validation, and 15% test. Historical entity features use only prior rows. Random row splitting is prohibited because it can place future behavior in training features.
