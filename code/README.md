# Support Agent

Terminal-based support triage agent for the HackerRank Orchestrate hackathon.

## Approach

The agent uses only the local `data/` corpus. It loads markdown/text/html/csv
files, splits them into deterministic chunks, retrieves relevant chunks with a
small TF-IDF style scorer, applies explicit risk/escalation rules, and writes the
required predictions CSV.

No live web calls or hardcoded secrets are used.

## Run

```powershell
py code\main.py
```

or:

```powershell
python code\main.py
```

## Output

The script reads:

```text
support_tickets/support_tickets.csv
```

and writes:

```text
support_tickets/output.csv
```

with the required columns:

```text
status, product_area, response, justification, request_type
```
