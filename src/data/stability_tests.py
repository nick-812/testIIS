
import pandas as pd
from sklearn.datasets import fetch_california_housing
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import *
from evidently.test_suite import TestSuite
from evidently.tests.base_test import generate_column_tests
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset
from evidently.tests import *


df = pd.read_csv("data/processed/current_data.csv", sep=",", header=0)
df1 = pd.read_csv("data/processed/reference_data.csv", sep=",", header=0)

data_integrity_dataset_tests = TestSuite(tests=[TestColumnsType(), TestNumberOfDriftedColumns()])
data_integrity_dataset_tests.run(reference_data=df1, current_data=df)
data_integrity_dataset_tests.save_html("reports/stability.html")


