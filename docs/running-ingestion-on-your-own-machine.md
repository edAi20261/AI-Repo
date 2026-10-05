# Running the ingestion on your own machine

This is for the person who will be loading the books. You do not need to know
Python, Docker or anything about the code. Follow the steps in order.

Set aside about an hour for the setup. You only do it once. After that, loading
a book is two commands.

---

## What you are actually doing

You put PDF books on your computer. A program reads every page with an AI,
breaks the text into pieces, and saves those pieces into our shared database so
students can ask questions about the book.

Your computer does the organising. The reading is done by Google's AI over the
internet, so a fast machine does not make it much faster — and the results go to
our shared database, not to your hard drive.

**A book of 300 to 400 pages takes about half an hour.** You can run two or three
books at the same time.

---

## Step 1 — Install Docker Desktop

Docker is what lets you run our program without installing Python or any of the
libraries it needs. It keeps everything in one sealed box.

1. Go to https://www.docker.com/products/docker-desktop/
2. Download the Windows version and install it.
3. When it asks about **WSL 2**, say yes. If it offers to install WSL for you,
   let it.
4. Restart the computer when it asks.
5. Open Docker Desktop and leave it running. You should see "Engine running" at
   the bottom left.

### Give Docker enough memory

This matters. The default is often too small and a book will die halfway.

1. Open Docker Desktop, click the **gear icon** (Settings).
2. Go to **Resources**.
3. Set **Memory** to at least **10 GB**.
4. Click **Apply & restart**.

---

## Step 2 — Install Git

Git is how you get a copy of our code.

1. Go to https://git-scm.com/download/win
2. Download and install. Accept all the default options.
3. When it finishes you will have a program called **Git Bash**. That is the
   black window you will type commands into from now on. Every command in this
   document goes in Git Bash.

---

## Step 3 — Get a copy of the code

Open **Git Bash** and type this, pressing Enter at the end:

```
cd ~
git clone https://github.com/edAi20261/AI-Repo.git
cd AI-Repo
```

It will ask for your GitHub username and password the first time. If you do not
have access to the repository yet, ask the AI team to add you.

You are now inside the project folder. Every command below assumes you are here.
If you close the window and come back later, type `cd ~/AI-Repo` first.

---

## Step 4 — Build the program

This downloads and prepares everything the program needs. It takes **15 to 30
minutes the first time** and you only ever do it once.

```
docker build -f Dockerfile.ingest -t edu-ai-ingest:local .
```

Leave it alone until it finishes. The last line should say `naming to ...
edu-ai-ingest:local`.

If you later get told the code has been updated, run `git pull` and then this
command again.

---

## Step 5 — Put in the keys

The program needs permission to reach our database and the AI. Those permissions
come as a set of keys from the AI team.

```
cp .env.ingest.example .env.ingest
```

Now open `.env.ingest` in Notepad and replace every `ASK` with the value the AI
team gives you. Do not change the lines that already have a value.

**These keys are live.** Do not put them in WhatsApp, do not email them, do not
paste them into a website, and do not commit them. The file is already set up so
git will ignore it.

---

## Step 6 — Check everything works

Before you spend hours reading a book, check the program can actually reach
everything.

```
docker run --rm --env-file .env.ingest edu-ai-ingest:local -m scripts.preflight_ingest
```

You want five lines all starting with `ok`:

```
  ok    database  ...
  ok    vectors   ...
  ok    vision    ...
  ok    storage   ...
  ok    embedder  ...
```

If any line says `FAIL`, stop and send the whole output to the AI team. Do not
start a book until all five pass — a book that cannot save its results is four
hours of paid AI reading thrown away.

---

## Step 7 — Prepare the books

Make a folder for the PDFs, for example `C:\books`. In Git Bash that folder is
written `/c/books`.

Then make a list of what to load. Create a file called `books.tsv` inside
`~/AI-Repo` with one line per book. Each line has **four things separated by a
Tab key** (not spaces):

```
the PDF file name <Tab> lecture ID <Tab> subject ID <Tab> the book title
```

For example:

```
science-grade7.pdf	9d3f1b20-0003-4a00-9000-000000000005	7b1e5c40-0a6e-4d1a-9f3c-5d2a91c00007	Science — Grade 7 — Term 1
```

The two ID codes come from the AI team. Ask for one pair per book. Getting them
wrong means the book is saved under the wrong lesson, so check them.

> Use Notepad, not Word. Word replaces straight quotes and can break the file.

---

## Step 8 — Load the books

```
INGEST_BOOKS_DIR=/c/books scripts/ingest_local.sh books.tsv
```

That is it. It checks the keys again, then works through the books one at a
time, printing a line as each one starts and finishes. Full details go into a
folder called `ingest-logs`.

**While it runs:**

- Leave the window open. Closing it stops the work.
- Make sure the laptop does not go to sleep. Windows Settings → System → Power →
  set "When plugged in, turn off my screen after" to Never.
- You can use the computer normally for other things.

To run two books at the same time instead of one:

```
INGEST_AT_ONCE=2 INGEST_BOOKS_DIR=/c/books scripts/ingest_local.sh books.tsv
```

Do not go above three. The limit is not your computer, it is how many requests
Google will accept from us at once.

---

## Step 9 — Check the result

```
docker run --rm --env-file .env.ingest -v "$PWD:/w" edu-ai-ingest:local -m scripts.ingest_status /w/books.tsv
```

For each book you get:

```
Science — Grade 7 — Term 1
    pages read   412
    chunks       1180
    status       done, students can use it
```

**`done, students can use it`** — finished, nothing more to do.

**`HELD BACK, too many pages could not be read`** — the book was read but too
many pages came out unreadable, so it was not given to students. Send the book
title to the AI team. Usually the PDF is a poor scan.

**`not finished`** — it stopped early. Run the same command from Step 8 again.

---

## If something goes wrong

**Run it again.** This is the most useful thing to know: pages that were already
read are remembered, so running the same command again carries on from where it
stopped instead of starting over. It does not cost anything extra and it does
not damage anything.

**"Cannot connect to the Docker daemon"** — Docker Desktop is not running. Open
it and wait for "Engine running".

**The run stops by itself and says nothing useful** — usually the laptop slept,
or Docker ran out of memory. Check Step 1's memory setting, then run it again.

**Anything else** — send the AI team the contents of the matching file in
`ingest-logs`. The last 30 lines are usually enough.

---

## Things worth knowing

**You cannot break the shared database by running this twice.** Loading the same
book again replaces that book's own data and touches nothing else.

**The AI reading costs money per page**, about 4 to 5 dollars for a 400-page
book, and it is charged to our account wherever the program runs. That is why
running the same book over and over is worth avoiding — but a run that stopped
and continues is free, because it does not re-read.

**You will not see progress for the first few minutes.** The program reads a
large batch of pages before it starts reporting. That is normal.
