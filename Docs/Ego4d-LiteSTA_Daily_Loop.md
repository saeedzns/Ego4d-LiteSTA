
# Ego4d-LiteSTA — Daily Colab ↔ GitHub ↔ VS Code Loop

This doc explains the **daily workflow** to run everything in **Colab**, log results, push artifacts to **GitHub**, review/fix in **VS Code**, and iterate.

---

## 0) One-time setup

### A) Deploy key for Colab
1. In Colab, generate a key pair that persists in Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !mkdir -p /content/drive/MyDrive/.ssh && chmod 700 /content/drive/MyDrive/.ssh
   !ssh-keygen -t ed25519 -f /content/drive/MyDrive/.ssh/colab_egolite -N "" -C "colab-ego4d-litesta"
   !cat /content/drive/MyDrive/.ssh/colab_egolite.pub
   ```
2. Copy the **public key** and add it to your GitHub repo (**Settings → Deploy Keys → Add deploy key**).
   - Title: `colab-ego4d-litesta`
   - ✅ **Allow write access**

### B) Set your GitHub user in notebooks
In the **Daily Repo Loop** cell of each notebook, set:
```python
GITHUB_USER = "your-gh-username"
REPO_NAME   = "Ego4d-LiteSTA"
```

---

## 1) What was added to your notebooks

At the top of **Ego4D_STA_Full_Colab.ipynb** and **Ego4D_STA_Colab_Starter.ipynb** we inserted cells that:
- Mount Google Drive.
- Configure SSH using `/MyDrive/.ssh/colab_egolite` (Deploy Key).
- Clone/pull your repo `Ego4d-LiteSTA` into `/content/Ego4d-LiteSTA`.
- Ensure folders: `runs/`, `logs/`, `outputs/`, `notebooks/`.
- Define a **run logger** with:
  - `RUN_ID` (timestamp, unique per run)
  - `log_write(msg)` → appends to `logs/<RUN_ID>.log`
  - `record_error(title, traceback)` → appends to `COLAB_ERRORS.md`

At the **end** of each notebook, we added a **Commit & Push** cell that stages `logs/*`, `runs/*`, `outputs/*`, and `COLAB_ERRORS.md`, then commits and pushes to GitHub.

We also generated a stand-alone **setup_colab.ipynb** that contains only the setup/pull/log/push loop.

---

## 2) Daily flow (Colab-first)

1. **Open** one of the notebooks (or `setup_colab.ipynb` + your task notebook).
2. **Run** the **Daily Repo Loop** setup cells to mount Drive, configure SSH, and pull the repo.
3. **Run your pipeline cells** (data, training, eval). Use the logger helpers.
4. **Export metrics & artifacts** into the repo:
   - `logs/<RUN_ID>.log` → plain-text log file.
   - `runs/<RUN_ID>/metrics.json` → small JSON summary (create this in your code).
   - `runs/<RUN_ID>/sta_val_preds.json` or other outputs.
   - On exceptions, call `record_error(...)` to append into `COLAB_ERRORS.md`.
5. **Run the Commit & Push** cell to push logs/artifacts back to GitHub.

**Template snippet for your train/eval cell:**
```python
import json, traceback, shutil

try:
    log_write("==> Starting training...")
    # ... your training code ...
    # Example metrics (replace with real values):
    metrics = {"selector_acc": 0.25, "verb_acc": 0.10, "ttc_mae": 1.40}
    with open(f"{RUN_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log_write(f"Metrics: {metrics}")

    # Example export of predictions:
    # shutil.copy("/content/sta_val_preds.json", f"{RUN_DIR}/sta_val_preds.json")

except Exception as e:
    import traceback
    record_error("Training crashed", traceback.format_exc())
    log_write("ERROR during training. See COLAB_ERRORS.md")
    raise
```

Then run the **Commit & Push** cell (already added).

---

## 3) VS Code loop (agent-assisted)

1. `git pull --rebase` to sync.
2. Review `COLAB_ERRORS.md` and the latest `runs/*/metrics.json`.
3. Ask your VS Code agent:
   > Read `COLAB_ERRORS.md` and the latest `runs/*/metrics.json`.  
   > 1) Fix failures and weak spots in `notebooks/` and Python modules.  
   > 2) Improve robustness for Colab (path checks, safe defaults).  
   > 3) Keep code lightweight.  
   > 4) Update comments/docs for any behavior change.
4. `git add -A && git commit -m "agent: fixes per COLAB_ERRORS + robustness updates" && git push`
5. Return to **Colab**, **pull**, and **run** again.

---

## 4) FAQ

### Q1) Will **COLAB_ERRORS.md** be created on each run?
- **It is created/updated only when you call `record_error(...)`.**  
  In the injected logger cell, `record_error` appends to `COLAB_ERRORS.md` whenever you catch an exception and pass its traceback. If your run has **no errors**, this file may remain unchanged.

### Q2) Will **runs/*/metrics.json** be created on each run?
- **Only if your code writes it.**  
  The notebook includes a template. You must save a `metrics.json` file yourself, e.g.:
  ```python
  with open(f"{RUN_DIR}/metrics.json", "w") as f:
      json.dump({"selector_acc": 0.25, "verb_acc": 0.10, "ttc_mae": 1.40}, f, indent=2)
  ```
  The **Commit & Push** cell will include it automatically once present.

### Q3) What is always created?
- `logs/<RUN_ID>.log` and the folder `runs/<RUN_ID>/` are created at the start of each run by the logger cell.

### Q4) Where do artifacts go?
- Put any outputs under `runs/<RUN_ID>/` or `outputs/`. The push cell stages both paths.

### Q5) Can I use my Windows SSH key?
- Use the **Deploy Key** approach for Colab (recommended). Your Windows SSH key stays for local VS Code. Deploy keys keep Colab secure and independent.
