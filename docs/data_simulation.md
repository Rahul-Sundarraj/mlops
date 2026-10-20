# Simulated delay data (16 trains)

Generate: `python -m src.data.simulate --start 2024-01-01 --end 2025-12-31 --seed 42`
Outputs (git-ignored, to be DVC-tracked in Phase 2): `data/simulated/delays.csv` (per train, day, station) and `grievances.csv` (per train-day).

Model: delay carries over between stops with partial recovery (higher for Vande Bharat), plus per-segment shocks scaled by
rain (weather proxy, region/month based: NE monsoon Oct-Dec TN, SW monsoon Jun-Aug Kerala/Karnataka coast), festival windows (x1.35),
Fri/Sun (x1.1), and rare incidents (3%/run, +30-120 min). Final-stop mean delay is calibrated near 8 / 25 / 32 min
(Vande Bharat / superfast / express); tune against Rajya Sabha punctuality figures.

Routes (stops, km, departure, direction) were checked on 2026-10-20 against confirmtkt/railmitra/ixigo timetables, not NTES.
16159/16160 are MS-MAQ Express (Chennai Egmore-Mangaluru), not Puducherry. Long trains list major stops only; 12679 stops are mirrored from 12680. Re-check against NTES before the final report.
