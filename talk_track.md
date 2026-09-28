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

**SHOW:** The table - 10,000 past loans with features (credit score, income,
debt-to-income, prior delinquencies, etc.) and a `defaulted` column.

**SAY:** "This is history: 10,000 loans we already know the outcome of. The model learns
which patterns tend to precede a default. No black box - it is learning from your data."

**CLICK:** Run (or scroll through) the training cells - point out the 3 runs with
different settings.

**SHOW:** MLflow auto-logging - each run captures its parameters, its accuracy metrics
(AUC), and the model artifact.

**SAY:** "Every experiment we run is captured automatically. Nothing is lost. This is the
lab notebook an examiner or a model-risk team would ask for - fully reproducible."

**CLICK:** Open the `todaybank-loan-default` experiment (left nav > Experiments).

**SHOW:** The 3 runs side by side; sort by AUC; highlight the best run.

**SAY:** "Here are our three attempts, compared on one screen. We pick the best performer -
this one - as our candidate. That comparison is the heart of experiment tracking."

---

## 02 - WALK: One governed model registry (7 min)  [deck: "02 - WALK" divider]

**CLICK:** In Catalog Explorer, open `todaybank_mlflow101.models.loan_default_risk`.

**SHOW:** The registered model, version 1, with the `@champion` alias.

**SAY:** "The winning model is now a governed asset in Unity Catalog - the same place your
data lives, with the same permissions and audit. `@champion` is a friendly label pointing
at the version that is 'in production'. Promoting a new model later is just moving that
label - no code change, no endpoint rebuild."

**CLICK:** Open the model's Lineage tab.

**SHOW:** Lineage from `lending.loan_applications` > the model.

**SAY:** "One click traces this model back to the exact table it learned from. If someone
asks 'what data trained the model deciding our loans?', that is the answer, on screen. That
is the governance story a bank needs from day one."

---

## 03 - RUN: Serve it, score it, watch it (8 min)  [deck: "03 - RUN" divider]

**CLICK:** Open the `todaybank-loan-default` serving endpoint. Show state READY.

**SHOW:** The endpoint page - it is a live REST API backed by the `@champion` model.

**SAY:** "The model is now a production API any system can call - the loan-origination
system, a dashboard, a batch job. It is no longer a notebook; it is a managed, versioned
service that scales up on demand and to zero when idle."

**CLICK:** Use the endpoint's Query panel (or notebook 03) to score two applicants live.

**SHOW:** Strong applicant (credit 780, low DTI, no delinquencies) > **PD 0.60%**.
Then risky applicant (credit 540, high DTI, 3 delinquencies) > **PD 7.93%**.

**SAY:** "Same model, two applicants, scored in real time. The strong applicant comes back
at about half a percent probability of default; the risky one at nearly eight percent -
over ten times higher. That number is what a credit decision or a risk-based price can be
built on."

**CLICK:** Open `lending.loan_applications_scored`.

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
