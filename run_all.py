"""Run the full experimental pipeline in order.

Total runtime is roughly 60 minutes (E3 dominates); each step resumes
where it left off, so the script can be interrupted and re-run.
"""
import runpy
import sys

STEPS = [
    'exp1_margin_validation.py',   # E1  (~15 min)
    'exp2_bounds_comparison.py',   # E2  (~5 min)
    'exp3_realdata.py',            # E3  (~45 min)
    'make_figures.py',
    'make_tables.py',
    'make_realdata_outputs.py',
]

if __name__ == '__main__':
    for step in STEPS:
        print(f"\n{'='*30} {step} {'='*30}", flush=True)
        sys.argv = [step]
        runpy.run_path(step, run_name='__main__')
    print("\nAll done. Outputs: results/, figures/, tables/.")
