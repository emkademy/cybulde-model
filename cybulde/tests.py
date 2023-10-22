from cybulde.utils.mlflow_utils import get_all_experiment_ids, get_best_run

experiments = get_all_experiment_ids()

print(f"{experiments=}")

best_runs = get_best_run()

print(f"{best_runs=}")
print(f"{best_runs['metrics.test_f1_score']=}")
