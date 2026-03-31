"""Patch row 144 in benchmark_v1.parquet with real Q&A (replaces mock)."""

import json
import pandas as pd
from pathlib import Path

PARQUET_PATH = Path("data/benchmarks/benchmark_v1.parquet")

# Real Q&A generated from the Northwind T-SQL setup script (Microsoft, 1994-2000)
QUESTIONS = [
    "What database is created by this SQL script?",
    "Which company holds the copyright for this script, and what years are listed?",
    "What file names are used for the database data file and log file?",
    "What DATEFORMAT is set in the script and why?",
    "What SQL Server major version threshold triggers different database option commands?",
    "What ALTER DATABASE command is used for SQL Server 12 and above?",
    "What system table and conditions are queried to determine the master database file directory?",
    "What three stored procedures are dropped if they exist before recreation?",
    "What SQL command is used to conditionally drop the Northwind database if it already exists?",
    "What views related to sales summaries are dropped at the beginning of the script?",
]

ANSWERS = [
    "The Northwind database.",
    "Microsoft, Inc. holds the copyright, with years 1994 - 2000.",
    "The data file is named 'northwnd.mdf' and the log file is named 'northwnd.ldf'.",
    "The DATEFORMAT is set to 'mdy' so that date strings are interpreted correctly regardless of the default DATEFORMAT on the server.",
    "SQL Server major version 12 (ProductMajorVersion < 12 triggers legacy sp_dboption calls).",
    "ALTER DATABASE [Northwind] SET RECOVERY SIMPLE WITH NO_WAIT",
    "master.dbo.sysaltfiles is queried WHERE dbid = 1 AND fileid = 1 to find the master.mdf path.",
    "The stored procedures dropped are: 'Employee Sales by Country', 'Sales by Year', and 'Ten Most Expensive Products'.",
    "IF EXISTS (SELECT * FROM sysdatabases WHERE name='Northwind') DROP DATABASE Northwind",
    "The views dropped include: 'Category Sales for 1997', 'Sales by Category', 'Sales Totals by Amount', 'Summary of Sales by Quarter', and 'Summary of Sales by Year'.",
]


def main():
    df = pd.read_parquet(PARQUET_PATH)
    print(f"Loaded {len(df)} rows from {PARQUET_PATH}")

    # Verify old mock answers before patching
    old_qs = json.loads(df.iloc[144]["questions"])
    print(f"Old Q1 (should be mock): {old_qs[0]}")

    df.at[df.index[144], "questions"] = json.dumps(QUESTIONS)
    df.at[df.index[144], "reference_answers"] = json.dumps(ANSWERS)

    df.to_parquet(PARQUET_PATH, index=False)
    print(f"Patched row 144 and saved to {PARQUET_PATH}")

    # Verify
    df2 = pd.read_parquet(PARQUET_PATH)
    new_qs = json.loads(df2.iloc[144]["questions"])
    new_ans = json.loads(df2.iloc[144]["reference_answers"])
    print("\nVerification:")
    for i, (q, a) in enumerate(zip(new_qs, new_ans)):
        print(f"  Q{i+1}: {q}")
        print(f"  A{i+1}: {a}")
        print()


if __name__ == "__main__":
    main()
