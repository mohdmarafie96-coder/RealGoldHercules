# Infrastructure setup: Google Cloud Storage and Compute Engine

Written for someone who has never used Google Cloud. Each step says whether it
can be done **in the web console (tablet-friendly)** or **needs a terminal**.

The short version: everything in Part A can be done from a tablet in the
browser. Part B has a console path too, but one command in Cloud Shell is
easier. Cloud Shell is a terminal inside the browser, so it works on a tablet.

**You never need to download a key file.** The VM gets its permissions from an
attached service account. Downloading a JSON key on a tablet is awkward and less
secure, so the instructions avoid it entirely. Part A.6 covers the key only for
the case where you want to run the ingest on your own laptop instead.

---

## Part A: Storage

### A.1 Create a project — console

1. Go to `console.cloud.google.com`.
2. Top bar, click the project dropdown, then **New Project**.
3. Name it something like `xau-research`. Note the **Project ID** that is
   generated (it looks like `xau-research-412345`); you will need it later.
4. Click **Create**.

### A.2 Enable billing — console

A card must be on file even though this setup is designed to sit inside the free
tier. Navigation menu, **Billing**, then link a billing account to the project.

New accounts normally get a free trial credit, which more than covers this.

### A.3 Pick a region and stick to it — decision, not a step

Choose **one** region and use it for both the bucket and the VM. If they differ,
you pay for network egress between them and the job runs slower.

Use `us-central1` unless you have a reason not to. It is one of the three
regions where the free-tier VM is available, and it is cheap.

Write it down. Every later step must match.

### A.4 Create the bucket — console

1. Navigation menu, **Cloud Storage**, **Buckets**, **Create**.
2. **Name**: must be globally unique across all of Google Cloud. Something like
   `xau-data-<your-initials>-2026`. Note it down.
3. **Location type**: Region. **Location**: `us-central1` (must match A.3).
4. **Storage class**: Standard.
5. **Access control**: Uniform. Leave *Enforce public access prevention* ticked.
6. **Protection**: None needed. Soft delete defaults are fine.
7. Create.

Cost note: the M1 history is about 60 MB. Standard storage is roughly $0.02 per
GB per month, so this bucket will cost well under a cent a month. The tick slice
would add roughly 250 MB, still trivial.

### A.5 Create the service account — console

A service account is a robot user. The VM will act as this robot.

1. Navigation menu, **IAM & Admin**, **Service Accounts**, **Create service
   account**.
2. **Name**: `xau-ingest`. The email is generated as
   `xau-ingest@<PROJECT_ID>.iam.gserviceaccount.com`. Note it down.
3. On the "Grant this service account access to project" step, **skip it**.
   Granting at project level is broader than needed. You will grant access on
   the bucket only, in the next step.
4. Done.

Now grant it access to just that bucket:

1. **Cloud Storage**, **Buckets**, click your bucket.
2. **Permissions** tab, **Grant access**.
3. **New principals**: paste the service account email from step 2.
4. **Role**: `Storage Object Admin`. This allows reading, writing and deleting
   objects in this bucket, and nothing else anywhere.
5. Save.

Why not `Storage Admin`: that role can also delete the bucket itself and change
its configuration. The ingest never needs to.

### A.6 Only if running outside Google Cloud — console plus a file transfer

Skip this if you are using the VM in Part B. The VM does not need a key.

1. Service Accounts, click `xau-ingest`, **Keys** tab, **Add key**, **Create new
   key**, JSON. The file downloads.
2. Put it somewhere private on the machine that will run the ingest, for example
   `~/.config/gcloud/xau-ingest.json`, and make it readable only by you
   (`chmod 600`).
3. Set `GOOGLE_APPLICATION_CREDENTIALS` to that path.

Treat that file like a password. Anyone holding it can read and write your
bucket. If it leaks, delete the key in the same **Keys** tab and create a new one.

### A.7 The environment variables

The code reads exactly two variables and has no bucket name or credential path
compiled into it.

| Variable | Value | Needed where |
|---|---|---|
| `XAU_DATA_URI` | `gs://<your-bucket-name>/xau` | Always, to write to GCS |
| `GOOGLE_APPLICATION_CREDENTIALS` | path to the JSON key | Only outside Google Cloud |

On the VM, only `XAU_DATA_URI` is set. Credentials come from the attached
service account automatically.

If `XAU_DATA_URI` is unset, the code writes to the local `data_root` from
`configs/config.yaml`, which is what it has been doing so far.

### A.8 DuckDB reading directly from the bucket

Two routes, because DuckDB and Google authenticate differently.

**From Python, using the same credentials as the ingest.** Nothing extra to
create:

```python
import duckdb, gcsfs
con = duckdb.connect()
con.register_filesystem(gcsfs.GCSFileSystem())     # uses ADC
con.sql("SELECT count(*) FROM read_parquet('gs://BUCKET/xau/bars_m1/**/*.parquet')")
```

**From the DuckDB CLI or SQL tools**, you need an HMAC key, which is a
Google-issued pair that makes the bucket look S3-compatible:

1. **Cloud Storage**, **Settings**, **Interoperability** tab.
2. Under "Service account HMAC keys", **Create a key for a service account**,
   pick `xau-ingest`.
3. You get an **Access key** and a **Secret**. The secret is shown once. Store it.

```sql
INSTALL httpfs; LOAD httpfs;
CREATE SECRET (TYPE GCS, KEY_ID 'GOOG...', SECRET '...');
SELECT count(*) FROM read_parquet('gs://BUCKET/xau/bars_m1/**/*.parquet');
```

The Python route is simpler and avoids a second credential. Use HMAC only if you
want to query from a SQL client directly.

---

## Part B: The machine that runs the download

The ingest is network-bound and mostly idle, waiting out rate limits. It needs
almost no CPU or memory. What it needs is to stay alive for many hours.

### B.1 Which machine, and why

**`e2-micro` in `us-central1`.** It is inside the Google Cloud free tier: one
non-preemptible `e2-micro` per month in `us-central1`, `us-west1` or
`us-east1`, plus 30 GB of standard persistent disk. For this workload that means
the VM is free and you pay only the fraction of a cent for bucket storage.

Do **not** use a Spot or Preemptible VM. They are cheaper but Google reclaims
them with 30 seconds notice, which for a 13-hour job means constant restarts.
The job resumes correctly, but there is no saving worth the hassle when the
non-preemptible option is already free.

Sizing sanity check: the job holds at most a month of M1 bars in memory before
flushing, roughly 44,000 rows of 16 float columns, a few megabytes. The
`e2-micro`'s 1 GB is ample.

### B.2 Create the VM

**Console path (tablet-friendly):**

1. Navigation menu, **Compute Engine**, **VM instances**, **Create instance**.
   If prompted, enable the Compute Engine API and wait a minute.
2. **Name**: `xau-ingest`.
3. **Region**: `us-central1`. **Zone**: `us-central1-a`.
4. **Machine configuration**: series `E2`, machine type `e2-micro`.
5. **Boot disk**: Change, select **Debian GNU/Linux 12 (bookworm)**, size 30 GB,
   type Balanced persistent disk.
6. **Identity and API access**: under **Service account**, select
   `xau-ingest@<PROJECT_ID>.iam.gserviceaccount.com`. This is the step that
   removes any need for a key file. Leave access scopes at default.
7. **Firewall**: leave both HTTP boxes unticked. Nothing needs to reach in.
8. Create.

**Cloud Shell path (one command, also works on a tablet).** Click the terminal
icon in the top bar of the console to open Cloud Shell, then:

```bash
gcloud compute instances create xau-ingest \
  --project=PROJECT_ID \
  --zone=us-central1-a \
  --machine-type=e2-micro \
  --image-family=debian-12 --image-project=debian-cloud \
  --boot-disk-size=30GB --boot-disk-type=pd-balanced \
  --service-account=xau-ingest@PROJECT_ID.iam.gserviceaccount.com \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --no-address-assign-external-ip-on-creation 2>/dev/null || \
gcloud compute instances create xau-ingest \
  --project=PROJECT_ID \
  --zone=us-central1-a \
  --machine-type=e2-micro \
  --image-family=debian-12 --image-project=debian-cloud \
  --boot-disk-size=30GB --boot-disk-type=pd-balanced \
  --service-account=xau-ingest@PROJECT_ID.iam.gserviceaccount.com \
  --scopes=https://www.googleapis.com/auth/cloud-platform
```

Replace `PROJECT_ID` in both places. The second form is the fallback if your
project policy requires an external IP.

### B.3 Connect

In the console, **Compute Engine**, **VM instances**, click **SSH** next to
`xau-ingest`. A terminal opens in the browser. This works on a tablet.

### B.4 Install

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip git

git clone https://github.com/mohdmarafie96-coder/RealGoldHercules.git
cd RealGoldHercules
git checkout claude/xauusd-data-layer-6d9eg8

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install gcsfs           # only needed for the DuckDB-from-Python route
```

Verify the bucket is reachable before starting a long job:

```bash
export XAU_DATA_URI=gs://YOUR-BUCKET/xau
python3 -c "
from xau_data.storage.store import open_store
s = open_store('./data')
print('store:', s.uri)
print('days already complete:', len(s.completed_days('XAUUSD')))
"
```

If that prints your `gs://` URI and a day count without an error, credentials
and permissions are correct. If it raises a permission error, revisit A.5.

### B.5 Run it so it survives disconnection

**systemd is the right answer** here, not `nohup`. `nohup` survives you closing
the browser tab, but not a VM reboot, and it gives you no restart-on-failure.
systemd gives both, and `journalctl` for logs.

```bash
sudo tee /etc/systemd/system/xau-ingest.service >/dev/null <<'UNIT'
[Unit]
Description=XAUUSD M1 candle backfill
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=YOUR_LINUX_USER
WorkingDirectory=/home/YOUR_LINUX_USER/RealGoldHercules
Environment=XAU_DATA_URI=gs://YOUR-BUCKET/xau
Environment=XAU_START=2021-01-01
ExecStart=/home/YOUR_LINUX_USER/RealGoldHercules/.venv/bin/python scripts_backfill_m1.py
Restart=on-failure
RestartSec=120
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable --now xau-ingest
```

Replace `YOUR_LINUX_USER` (run `whoami`) and `YOUR-BUCKET`.

Watch it:

```bash
journalctl -u xau-ingest -f          # live
journalctl -u xau-ingest | tail -50  # recent
systemctl status xau-ingest          # is it alive
```

Stop, restart, and check progress without stopping:

```bash
sudo systemctl stop xau-ingest
sudo systemctl start xau-ingest
python3 -c "
from xau_data.storage.store import open_store
print(len(open_store('./data').completed_days('XAUUSD')), 'days done')
"
```

Killing it at any moment is safe. Resume is decided by asking the bucket what
already exists, so a restart refetches nothing.

If you prefer `nohup` anyway:

```bash
cd ~/RealGoldHercules && source .venv/bin/activate
export XAU_DATA_URI=gs://YOUR-BUCKET/xau
nohup python3 scripts_backfill_m1.py > ~/m1.log 2>&1 &
disown
tail -f ~/m1.log
```

### B.6 When it finishes

The log ends with a line beginning `BACKFILL_M1_DONE` followed by JSON. Then:

```bash
sudo systemctl disable --now xau-ingest      # stop it restarting
```

And delete the VM to be certain it is not billing, from the console
(**VM instances**, tick it, **Delete**) or:

```bash
gcloud compute instances delete xau-ingest --zone=us-central1-a
```

The data is in the bucket, not on the VM, so deleting the VM loses nothing.

---

## What you need to give me back

Once Part A is done, I only need the bucket URI:

```
XAU_DATA_URI=gs://your-bucket-name/xau
```

No credentials. Do not paste a service account key or an HMAC secret into chat.

---

## Cost summary

| Item | Monthly |
|---|---|
| `e2-micro` in `us-central1` | $0 within free tier |
| 30 GB standard persistent disk | $0 within free tier |
| Bucket storage, ~60 MB M1 history | under $0.01 |
| Bucket storage, ~310 MB with tick slice | about $0.01 |
| Egress to the internet | $0, the job only downloads |

The realistic bill is a rounding error, provided the VM is `e2-micro` in one of
the three free-tier regions and you delete it when the backfill is done.
