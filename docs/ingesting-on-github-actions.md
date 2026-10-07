# Ingesting a book on GitHub's machines

This is the same ingestion the laptop guide describes, run by GitHub instead of
your computer. You give it a link to the PDF and the book's IDs; GitHub's
machines read, embed and index it; your computer can be closed.

The only thing that costs money is what it costs now: Google reading the pages.
GitHub's machines are free within the repository's monthly Actions minutes.

---

## Why it is faster

A scanned book is two kinds of slow work:

- **Reading pages.** Every scanned page is two calls to Google's model — one for
  its layout, one for its words — at about 12.6 seconds each. One machine can
  only keep so many in flight.
- **Embedding chunks.** Turning each chunk of text into a vector for search.
  Measured at 2.5 seconds a chunk on a 2-core machine: about 25 minutes for a
  book of 600 chunks, on one machine.

Neither needs the others to finish, so the workflow splits both across
machines: by default six machines read the pages and eight embed the chunks.

The pipeline itself was also changed, which helps on a laptop too:

| | Before | Now |
|---|---|---|
| Renders per scanned page | 2, one at a time on one core | 1, on every core |
| Rendering a page (300 dpi, 2 cores) | 0.83 s, twice | 0.45 s, once |
| Model calls in flight on one machine | 8 | 16 (8 per GitHub machine) |
| A re-run of a book already read | renders every page again | renders only pages with figures to cut |

---

## One-time setup

Someone with admin access to the repository does this once.

### 1. Add the secrets

Repository → **Settings** → **Secrets and variables** → **Actions** →
**Secrets** → **New repository secret**. The names, and the same values as
`.env.ingest`:

| Secret | What it is |
|---|---|
| `DATABASE_URL` | the database the chunks and the page cache go into |
| `QDRANT_URL` | the search index |
| `QDRANT_API_KEY` | its key |
| `S3_BUCKET` | where figure crops are stored |
| `S3_REGION` | e.g. `eu-central-1` |
| `AWS_ACCESS_KEY_ID` | for S3 |
| `AWS_SECRET_ACCESS_KEY` | for S3 |
| `GOOGLE_APPLICATION_CREDENTIALS` | the service account's whole JSON file, pasted on one line |

Secrets are never shown in logs, and nobody can read them back once saved.

### 2. Choose which index it writes to

Runs write to production by default. To point them somewhere else, add a
**variable** (same page, **Variables** tab) called `INGEST_APP_ENV` — for
example `local` writes to the development collections.

### 3. Let GitHub's machines reach the database

GitHub's machines have no fixed address. If the database only accepts
connections from listed IP addresses, the first step of every run will fail
with `FAIL database`. Either allow connections from anywhere (it still needs the
password, over SSL), or run the workflow on a machine you control (a GitHub
"self-hosted runner") that is on the list.

---

## Ingesting a book

### From the website

1. Upload the PDF to Google Drive. **Share** → **General access** → **Anyone
   with the link**. Copy the link.
2. In the repository: **Actions** → **Ingest a book** (left side) → **Run
   workflow**.
3. Paste the link, and the ID of the subject the book belongs to
   (`academics.subjects.id`). A run whose subject does not exist stops on its
   first step and says so. Tick **Preview only** to see the lessons the book
   will be split into, on the run's summary page, without writing anything;
   run it again unticked to ingest. The second run reuses every page the first
   one read.
4. **Run workflow**. It appears in the list within a few seconds.

Everything else comes from the subject or the book:

- **Language** — from the words on its pages, once they are read. A book is
  Arabic when most of its letters are, so an English workbook with Arabic
  glosses stays English.
- **Lessons** — the book is split into the lessons it teaches, from its table
  of contents and the headings on its pages. Each lesson becomes its own row in
  `content.lessons` under the subject (as a draft), and every chunk, figure and
  topic is filed under the lesson its page belongs to. The run's summary lists
  each lesson with its ID.
- **Grade** — from the subject.
- **Book ID** — from the file's bytes. Uploading the same PDF again updates the
  same lessons rather than adding a second set; lessons a new split no longer
  has are archived.

When it finishes, the summary at the top of the run shows the status, the IDs
it used, pages, unreadable pages, figures, chunks, topics and prerequisite
links.

### Several books at once

With the GitHub CLI installed and logged in (`gh auth login`), put one book per
line in `books.txt` — its link, a space, and its subject ID — and run:

```
scripts/ingest_on_github.sh books.txt
```

Each book is its own run, and they run side by side. Your computer only sends
the request.

---

## What a run does

```
plan ──► read × 6 ──► prepare ──► embed × 8 ──► finish
warm-model ─────────────────────────┘
```

| Step | What happens |
|---|---|
| plan | decides the machines, from the inputs — seconds |
| read | each machine checks every credential, fetches the book and reads its share of the pages into the page cache |
| warm-model | gets the embedding model ready while the pages are being read |
| prepare | extracts and chunks with every page already read; cuts and captions figures |
| embed | each machine embeds its share of the chunks |
| finish | topics, prerequisites, and writing the index |

**Nothing is written to the index until the last step.** A run that fails
part way leaves the index as it was.

**A failed run is safe to re-run.** Every page already read is kept, so a
second run only pays for pages that were never read.

---

## How long and what it costs

These are estimates from measured parts, not from a whole run timed end to end.
Time the first real runs and correct this table.

For a 300-page scanned book of about 600 chunks:

| | |
|---|---|
| Reading the pages | about 3–5 minutes |
| Embedding | about 4–5 minutes |
| Whole run, start to summary | about 15–20 minutes |
| The very first run | a few minutes more, while GitHub caches the Python packages and the model |
| Google | about $0.005–$0.0065 a page (the repository's own measurement): about **$2 a book** |
| GitHub | about 75 machine-minutes a book |

**Why not 10 minutes:** each of the five steps waits for the one before, and
each GitHub machine spends about a minute starting up before it does any work.
On free 2-core machines that start-up time, plus embedding on CPU, is the
floor. Getting under 10 minutes needs one of: a GPU for embedding, larger
(paid) GitHub machines, or a public repository (whose free machines have 4
cores).

**Minutes:** a private repository on GitHub's free plan has a monthly allowance
of Actions minutes — check **Settings → Billing** for yours. At about 75 minutes
a book, 2,000 minutes is about 25 books a month.

---

## When something goes wrong

**`FAIL database` / `FAIL vectors` / `FAIL vision` / `FAIL storage`** in the
first step — that credential is missing or wrong, or (database) GitHub's
machines are not allowed to connect. Nothing was paid for.

**"what was downloaded is not a PDF … share the file as 'Anyone with the
link'"** — the Drive file is private. Change its sharing and re-run.

**Many "429" or "rate limited" lines in a read step** — Google is turning
requests away. Re-run with fewer read machines (for example 4): every page
already read is kept.

**Status `needs_review`** — more than 5% of the pages could not be read. The
summary says how many. Re-running reads only those pages again.

---

## Before you ingest a book

Check its copyright page. A book indexed here is reproduced into a retrieval
system, and many publishers forbid exactly that without written permission. A
copy downloaded from a file-sharing channel is not a licensed copy.
