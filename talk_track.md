# TodayBank Session 6 - MLflow & MLOps 101: Talk Track

CLICK / SHOW / SAY cadence for the ~25-minute live showcase. Audience has no ML
experience - define every term in plain English before showing it. Formatting: no
em-dashes (en-dashes / hyphens); ASCII `>` and `<` for arrows.

**Live environment (verified):**
- Workspace: `https://adb-7405619514730982.2.azuredatabricks.net` (profile wb-genie-demo)
- Catalog `todaybank_mlflow101`, schemas `lending` + `models`
- Tables: `lending.loan_applications` (10,000 labeled), `lending.new_applications` (200),
  `lending.loan_applications_scored` (200 scored)
- Experiment: `/Users/duffy.walsh@databricks.com/todaybank-loan-default` (3 runs)
- Model: `todaybank_mlflow101.models.loan_default_risk` @champion = v1
- Endpoint: `todaybank-loan-default` (READY, scale-to-zero)
- Notebooks: `/Users/duffy.walsh@databricks.com/todaybank-mlflow-101/` (01-04)

## Pre-flight (do 5 minutes before the session)
- **Warm the endpoint** - it is scale-to-zero and cold-starts (~1 min) on first call.
  Send one scoring request before you present so the live call is instant on stage.
- Start the SQL warehouse (`22b7cf4bfffde7dc`) so table previews load fast.
- Open these browser tabs in order: (1) notebook 02, (2) the `todaybank-loan-default`
  experiment, (3) the model in Catalog Explorer, (4) the serving endpoint, (5)
  `lending.loan_applications_scored`.
- Have the deck open to the "00 - Demo" divider.

---

## 00 - Demo (2 min)  [deck: "00 - Demo" divider]

**SAY:** "We are going to follow one everyday bank decision - should we approve a loan,
and at what risk - all the way through the machine-learning lifecycle. ML just means
learning patterns from your own history to predict the next outcome. MLOps is how you do
that reliably and safely, the same way you run any production system. It is mostly live,
not slides."

**SAY (frame the arc):** "Three steps, same as the GenAI session: Crawl - train and track;
Walk - register and govern; Run - serve and monitor."

---

## 01 - CRAWL: Train once, track everything (8 min)  [deck: "01 - CRAWL" divider]

**CLICK:** Open notebook `02_train_track_register`. Scroll to the data preview of
`lending.loan_applications`.
  - **How:** Left nav > **Workspace** > `Users` > `duffy.walsh@databricks.com` > `todaybank-mlflow-101` > **`02_train_track_register`** (or top search bar / Cmd+P > type `02_train_track_register`). Attach compute via **Connect** (top-right). Scroll to **cell 5** - the `display(spark.table(...))` cell directly under the **`### Data preview`** header (2nd cell of Step 1) - and **execute it (Shift+Enter)** to render the scrollable 10,000-row grid.

**SHOW:** The table - 10,000 past loans with features (credit score, income,
debt-to-income, prior delinquencies, etc.) and a `defaulted` column.

**SAY:** "This is history: 10,000 loans we already know the outcome of. The model learns
which patterns tend to precede a default. No black box - it is learning from your data."

---

**CLICK:** Run (or scroll through) the training cells - point out the 3 runs with
different settings.
  - **How:** In the same notebook, scroll to **cell 12** - the training cell under the **`## Step 4 - Train 3 runs`** header (it starts with `# Set a named experiment`). First point at the **`CONFIGS`** list near the top of that cell (three `{...}` rows = 100 trees/depth 3, 200/depth 4, 150/depth 5), then **execute it (Shift+Enter)**. Output prints `--- Starting Run 1/2/3 ---` with an AUC each, ending in a `=== Run comparison ===` table. Note: re-running APPENDS 3 new runs to the experiment - to keep it at exactly 3, scroll instead of run.

**SHOW:** MLflow auto-logging - each run captures its parameters, its accuracy metrics
(AUC), and the model artifact.

**SAY:** "Every experiment we run is captured automatically. Nothing is lost. This is the
lab notebook an examiner or a model-risk team would ask for - fully reproducible."

---

**CLICK:** Open the `todaybank-loan-default` experiment (left nav > Experiments).
  - **How:** Click the **flask / Experiment icon** at the top-right of the notebook (jumps straight to this notebook's experiment); or left nav > **Experiments** > search **`todaybank-loan-default`**. In the runs table, click the **`auc_roc`** column header to sort, then select the top row.

**SHOW:** The 3 runs side by side; sort by AUC; highlight the best run.

**SAY:** "Here are our three attempts, compared on one screen. We pick the best performer -
this one - as our candidate. That comparison is the heart of experiment tracking.

auc_roc means Area Under the Receiver Operating Characteristic curve. It measures how well a binary-classification model separates two groups—for example:

fraudulent vs. legitimate transactions
sick vs. healthy patients
likely-to-churn vs. likely-to-stay customers
The key idea: it evaluates the model’s ranking ability, not just whether its final yes/no predictions are correct."

---

## 02 - WALK: One governed model registry (7 min)  [deck: "02 - WALK" divider]

**CLICK:** In Catalog Explorer, open `todaybank_mlflow101.models.loan_default_risk`.
  - **How:** Left nav > **Catalog** > expand **`todaybank_mlflow101`** > **`models`** schema > **Models** > **`loan_default_risk`** (or top search bar / Cmd+P > type `loan_default_risk` and pick the Model result).

**SHOW:** The registered model, version 1, with the `@champion` alias.

**SAY:** "The winning model is now a governed asset in Unity Catalog - the same place your
data lives, with the same permissions and audit. `@champion` is a friendly label pointing
at the version that is 'in production'. Promoting a new model later is just moving that
label - no code change, no endpoint rebuild."

---

**CLICK:** Open the model's Lineage tab.
  - **How:** On the model page, click the **`Lineage`** tab in the top tab row (next to Details / Versions). If lineage sits on the version, open **Version 1** first, then its **Lineage**, and click **`See lineage graph`** to show the `loan_applications` table > model arrow.

**SHOW:** Lineage from `lending.loan_applications` > the model.

**SAY:** "One click traces this model back to the exact table it learned from. If someone
asks 'what data trained the model deciding our loans?', that is the answer, on screen. That
is the governance story a bank needs from day one."

---

## 03 - RUN: Serve it, score it, watch it (8 min)  [deck: "03 - RUN" divider]

**CLICK:** Open the `todaybank-loan-default` serving endpoint. Show state READY.
  - **How:** Left nav > **Serving** > **`todaybank-loan-default`** (or top search bar / Cmd+P > type the name). Confirm the green **`Ready`** state at the top of the page.

**SHOW:** The endpoint page - it is a live REST API backed by the `@champion` model.

**SAY:** "The model is now a production API any system can call - the loan-origination
system, a dashboard, a batch job. It is no longer a notebook; it is a managed, versioned
service that scales up on demand and to zero when idle."

---

**CLICK:** Use the endpoint's Query panel (or notebook 03) to score two applicants live.
  - **How:** On the endpoint page, click **`Query endpoint`** (top-right). Paste the STRONG applicant JSON (see **Query payloads** at the end of this doc) into the request box > **Send** > read the `predictions` value (~0.006 = 0.60%). Then replace with the RISKY JSON > **Send** (~0.079 = 7.93%). (Fallback: in notebook `03_serve`, run **cell 10** - the Step 4 scoring cell - after first running setup cells 2, 4, 6, 8.)

**SHOW:** Strong applicant (credit 780, low DTI, no delinquencies) > **PD 0.60%**.
Then risky applicant (credit 540, high DTI, 3 delinquencies) > **PD 7.93%**.

**SAY:** "Same model, two applicants, scored in real time. The strong applicant comes back
at about half a percent probability of default; the risky one at nearly eight percent -
over ten times higher. That number is what a credit decision or a risk-based price can be
built on."

---

**CLICK:** Open `lending.loan_applications_scored`.
  - **How:** Left nav > **Catalog** > **`todaybank_mlflow101`** > **`lending`** > **`loan_applications_scored`** > **Sample Data** tab (or top search bar / Cmd+P > type `loan_applications_scored`). 200 rows, each with a probability-of-default score and a risk tier column.

**SHOW:** 200 fresh applications, each with a PD and a risk tier (Low/Medium/High).

**SAY:** "And in batch: 200 new applications scored and written back to a governed table,
ready for the business. Same model, both real-time and batch."

**SAY (monitoring, conceptual):** "The last part of MLOps is watching it. Because every
prediction and every access is logged in the platform, you can watch for drift - the world
changing so the model goes stale - and retrain, re-register, and flip the `@champion` label
when a better version is ready. That closes the loop."

---

## Close (1 min)  [deck: "Where a bank Gets Value"]

**SAY:** "That is the whole lifecycle on one governed platform: train and track, register
and govern, serve and monitor. Every experiment reproducible, one model registry in Unity
Catalog, real-time decisions behind a managed API, and the same pattern reused for the next
model - fraud, churn, marketing. Production-grade means governed, auditable, and yours."

**SAY (next step):** "The natural next step is a scoped pilot on your own data." Then open
for Q&A.

---

## If a live step misbehaves (fallbacks)
- **Endpoint slow on first call:** you forgot the warmup - keep talking about the `@champion`
  promote-by-alias story for ~30-60s while it scales up, then retry.
- **Warehouse cold:** preview a table you already opened in pre-flight instead.
- **Any live failure:** the deck dividers carry the narrative; the recorded PDs (0.60% vs
  7.93%) are in this doc to quote.


---

## Query payloads (copy/paste for the endpoint Query panel)

Request format: `{"dataframe_records": [ <applicant> ]}`

STRONG applicant (expect PD ~0.60%):
`{"dataframe_records":[{"credit_score":780,"annual_income":95000,"dti_ratio":14.5,"loan_amount":15000,"loan_term_months":36,"interest_rate":7.5,"employment_years":12.0,"num_prior_delinquencies":0,"home_ownership":"OWN","loan_purpose":"home_improvement"}]}`

RISKY applicant (expect PD ~7.93%):
`{"dataframe_records":[{"credit_score":540,"annual_income":32000,"dti_ratio":48.0,"loan_amount":25000,"loan_term_months":60,"interest_rate":21.0,"employment_years":0.5,"num_prior_delinquencies":3,"home_ownership":"RENT","loan_purpose":"debt_consolidation"}]}`
