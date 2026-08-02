import json
import wandb

# 1. Authenticate with your API key
wandb.login()

# Initialize the public API client
api = wandb.Api()

# 2. Reference your specific run: "entity_name/project_name/run_id"
# (Entity name is usually your username or team name)
run = api.run("/smahalanobis-uc-davis/hjepa-vwm/runs/3syv6wp2")

# 3. Pull the data fields according to W&B official attributes
run_data = {
    "run_id": run.id,
    "name": run.name,
    "notes": run.notes,
    "tags": run.tags,
    # run.config contains hyperparameters as a standard dict
    "config": run.config,
    # ._json_dict strips internal W&B wrapping from the metrics summary
    "summary": run.summary._json_dict,
    # .scan_history() downloads all logged steps (unlike .history() which samples)
    "history": [row for row in run.scan_history()],
}

# 4. Export everything cleanly to a local JSON file
with open("wandb_export.json", "w", encoding="utf-8") as f:
    json.dump(run_data, f, indent=4)

print("Data successfully exported to wandb_export.json")