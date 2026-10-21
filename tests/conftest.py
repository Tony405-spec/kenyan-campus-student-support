"""Small fixtures for repeatable unit tests."""

import pandas as pd
import pytest


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    """A valid miniature dataset with all three target classes."""
    return pd.DataFrame(
        {
            "Marital status": [1, 2, 1],
            "Application mode": [1, 2, 1],
            "Application order": [1, 2, 3],
            "Course": [10, 11, 10],
            "Daytime/evening attendance\t": [1, 1, 0],
            "Previous qualification": [1, 2, 1],
            "Previous qualification (grade)": [120.0, 130.0, 140.0],
            "Nacionality": [1, 1, 2],
            "Mother's qualification": [1, 2, 1],
            "Father's qualification": [1, 2, 1],
            "Mother's occupation": [1, 2, 1],
            "Father's occupation": [1, 2, 1],
            "Admission grade": [120.0, 130.0, 140.0],
            "Displaced": [1, 0, 1],
            "Educational special needs": [0, 0, 0],
            "Debtor": [0, 1, 0],
            "Tuition fees up to date": [1, 0, 1],
            "Gender": [0, 1, 0],
            "Scholarship holder": [1, 0, 1],
            "Age at enrollment": [20, 21, 22],
            "International": [0, 0, 1],
            "Curricular units 1st sem (credited)": [0, 1, 2],
            "Curricular units 1st sem (enrolled)": [6, 6, 6],
            "Curricular units 1st sem (evaluations)": [6, 5, 6],
            "Curricular units 1st sem (approved)": [5, 3, 6],
            "Curricular units 1st sem (grade)": [13.0, 10.0, 15.0],
            "Curricular units 1st sem (without evaluations)": [0, 1, 0],
            "Curricular units 2nd sem (credited)": [0, 0, 1],
            "Curricular units 2nd sem (enrolled)": [6, 6, 5],
            "Curricular units 2nd sem (evaluations)": [6, 5, 5],
            "Curricular units 2nd sem (approved)": [5, 2, 4],
            "Curricular units 2nd sem (grade)": [13.0, 9.0, 12.0],
            "Curricular units 2nd sem (without evaluations)": [0, 1, 0],
            "Unemployment rate": [10.0, 11.0, 9.0],
            "Inflation rate": [1.0, 2.0, 0.0],
            "GDP": [1.2, 1.1, 1.3],
            "Target": ["Dropout", "Enrolled", "Graduate"],
        }
    )
