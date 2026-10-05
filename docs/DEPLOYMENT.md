# Deploying the dashboard

The dashboard is a Streamlit app (`app/streamlit_app.py`). It only reads a small exported "bundle" of
results, so it needs no GPU, no training libraries and no access to the race data.

## 1. Put your results into the app

After the final evaluation has run (`python -m f1pit.evaluate`):

```bash
python -m f1pit.dashboard          # writes app/data/ from results/ and data/processed/
streamlit run app/streamlit_app.py # http://localhost:8501
```

The app shows, in this order of preference:

1. the folder named in the `F1PIT_DASHBOARD_DATA` environment variable (an error is shown if that
   folder holds no bundle; it does not fall back);
2. `app/data/` (your results);
3. `app/demo_data/` (computer-generated practice data, shown with a warning banner).

`app/data/` is about 1 to 3 MB. Commit it so the deployed app shows your results:

```bash
git add app/data && git commit -m "Dashboard data from the final evaluation" && git push
```

If you ran the pipeline on Colab, the notebook's last section builds `app/data` and downloads it as
`dashboard_data.zip`; unzip it into the `app/` folder of your repository.

## 2. Option A: Streamlit Community Cloud (free, quickest)

You need a free GitHub account. `app/data/` in this repository already holds the dashboard data of the
final evaluation, so the public app shows real results as soon as it is deployed.

**Step 1. Put the project on GitHub.** Create an empty repository at <https://github.com/new> (name it
`f1-pit-engine`, choose **Public**, do not add a README). Then, in the project folder:

```bash
git init -b main
git add .
git commit -m "F1 tyre-aware pit stop engine"
git remote add origin https://github.com/<your-username>/f1-pit-engine.git
git push -u origin main
```

GitHub asks for your user name and a **personal access token** in place of a password (create one at
GitHub > Settings > Developer settings > Personal access tokens, with the `repo` scope).

**Step 2. Check the tests on GitHub.** Open the repository's **Actions** tab. The CI workflow runs the
lint, the tests and the dashboard image build. Wait for the green tick.

**Step 3. Deploy.**

1. Sign in at <https://share.streamlit.io> with your GitHub account.
2. Choose **Create app**, then **Deploy a public app from GitHub**.
3. Repository: `<your-username>/f1-pit-engine`. Branch: `main`. Main file path: `app/streamlit_app.py`.
4. Optional: choose the address, for example `f1-pit-engine`, giving `https://f1-pit-engine.streamlit.app`.
5. In **Advanced settings** choose Python 3.11 or 3.12, then **Deploy**.

The first build takes a few minutes. Streamlit installs `app/requirements.txt` (the file next to the
app), which lists only the dashboard's own dependencies. Every push to `main` redeploys the app. A free
app goes to sleep after some days without visitors; anyone opening the link wakes it up.

**Step 4. Update the results later.** After a new final evaluation on your computer:

```bash
python -m f1pit.dashboard
git add app/data && git commit -m "Dashboard data from the final evaluation" && git push
```

## 3. Option B: Docker (any container host)

```bash
docker build -f Dockerfile.dashboard -t f1pit-dashboard .
docker run --rm -p 8501:8501 f1pit-dashboard      # http://localhost:8501
```

- The container listens on port **8501** and runs as a non-root user.
- Health check: `GET /_stcore/health` returns `ok`.
- To show a different bundle without rebuilding, mount it and point the app at it (the folder must
  already hold an exported bundle):

```bash
docker run --rm -p 8501:8501 -v "$PWD/app/data:/bundle:ro" -e F1PIT_DASHBOARD_DATA=/bundle f1pit-dashboard
```

The image runs on any service that hosts containers. Give the service port 8501 and the health path above.

## 4. Continuous integration and releases

| Workflow | When | What it does |
| --- | --- | --- |
| `.github/workflows/ci.yml` | every push and pull request | lint, tests, builds the test image, builds the dashboard image, runs the app script inside it, starts it and checks the health endpoint and the page |
| `.github/workflows/release.yml` | pushing a tag such as `v1.0.0` | runs the tests, then builds the dashboard image and publishes it to the GitHub Container Registry as `ghcr.io/<owner>/<repo>-dashboard` |

Release a version:

```bash
git tag v1.0.0 && git push origin v1.0.0
docker run --rm -p 8501:8501 ghcr.io/<owner>/<repo>-dashboard:1.0.0
```

Write `<owner>/<repo>` in lower case: image names cannot contain capital letters. A plain version
tag also updates `latest`; a pre-release tag such as `v1.0.0-rc1` does not. The first published
package is private; change its visibility in the package's own settings page on GitHub if others
should be able to pull it.

## What the app does not do

It does not download race data, train or score models, or make live predictions during a race. The
race replay steps through held-out races that the models have already scored. Until you tick the
"reveal" box it shows only what was known while the chosen lap was being driven: the table, the
timeline up to that lap, and stops already made. The race scorecard appears after the reveal.

If the results have a problem (models trained in quick mode, or the check that the models match the
data did not pass), a red notice is shown at the top of every page.
