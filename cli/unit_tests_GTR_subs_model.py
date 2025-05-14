#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 12:56:43 2025

@author: annabel
"""
import unittest
import pandas as pd
from datetime import datetime


# List of test files in desired order
ordered_test_files = [
    'tests/substitution_model_tests/test_subs_rate_matrix_construction.py',
    'tests/substitution_model_tests/test_conditional_prob_subs.py',
    'tests/substitution_model_tests/test_joint_prob_subs.py',
    'tests/substitution_model_tests/test_score_alignment.py',
    'tests/substitution_model_tests/test_alignment_loglike_GTR.py',
    'tests/substitution_model_tests/test_alignment_loglike_GTR_mixture.py',
    'tests/substitution_model_tests/test_GTR_mixture_degeneracy.py',
    'tests/substitution_model_tests/test_xrate_GTR_likelihood_match.py',
    'tests/substitution_model_tests/test_xrate_GTR_mixture_likelihood_match.py'
]

def load_and_run_tests(test_files):
    results_summary = []

    for test_file in test_files:
        suite = unittest.TestSuite()
        loader = unittest.TestLoader()

        # Convert file path to module path (replace / with . and drop .py)
        module_name = test_file.replace('/', '.').replace('\\', '.').replace('.py', '')

        try:
            suite.addTests(loader.loadTestsFromName(module_name))
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)

            status = 'PASS' if result.wasSuccessful() else 'FAIL'
            num_failures = len(result.failures) + len(result.errors)

        except Exception as e:
            print(f"Error loading or running {module_name}: {e}")
            status = 'ERROR'
            num_failures = None

        results_summary.append({
            'module': module_name,
            'status': status,
            'failures_or_errors': num_failures
        })

    # Convert to pandas DataFrame and print
    df = pd.DataFrame(results_summary)
    print("\nTest Summary:\n")
    print(df)
    
    with open(f'GTR-UNIT-TESTS.tsv','w') as g:
        timestamp = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
        g.write(f'# RUN ON: {timestamp}\n')
        g.write(f'#\n')
        df.to_csv(g, sep='\t')
    

if __name__ == '__main__':
    load_and_run_tests(ordered_test_files)


