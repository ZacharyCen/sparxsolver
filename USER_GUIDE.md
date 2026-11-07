# Sparx Solver User Guide

## 1. Start the application

On Windows, open `AutoAnswer_Win_x64.exe` if you have a packaged copy. To run from source, install Python 3.10 or later and run these commands in the project folder:

```bash
python -m venv .venv
```

Activate the environment with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on macOS/Linux, then run:

```bash
python -m pip install -r requirements.txt
python main.py
```

Desktop capture and input require a graphical desktop and the appropriate operating-system permissions. Separate-browser mode requires Chrome.

## 2. Configure the model connection

Create a private `OpenAI-KEY.txt` beside the executable, or in the project folder when running from source:

```text
API_KEY=YOUR_API_KEY_HERE
BASE_URL=https://api.openai.com/v1
MODEL=gpt-6.1-terra
```

Replace the placeholder with your own key. Your provider must support the selected model. Keep this file private and never upload it to GitHub. Environment variables `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL` are also supported; an existing key file can override them.

Open **Configuration → Model connection**, select **Remote API**, and check the server URL and model. Remote mode sends captured question images to the configured provider and may incur API charges.

## 3. Choose your homework

On **Solver**, choose one browser mode:

- **Separate browser:** click **Open separate browser · sign in once**, sign in to Sparx, and open your homework there. This mode leaves your desktop mouse and keyboard free.
- **Existing Chrome:** choose the homework window. Use **Refresh windows** if needed, or **Select screen area** for a custom capture region. This mode controls your desktop mouse and keyboard; avoid using them during a run.

Keep the question, answer controls, question tabs, and Summary visible. Avoid changing the browser page or window layout while solving.

Click **New set** and name the assignment. Use **Open saved set** only to resume that same homework. Different assignments need separate records, even when their question codes match.

## 4. Set options and run

- Enable **Bookwork check** when the assignment requires an end check. Disable it for exercises without one.
- Enable **Save workings** if you want a mathematical workings file.

With **Save workings** checked, files save automatically after each verified correct answer to **Documents → Sparx Solver → Workings → your assignment folder**. The Solver page shows the location; use **Workings → Open folder** to find the Markdown, equation images, and readable HTML copy. No Save dialog or final export is required. Unchecking it stops automatic updates and keeps existing files.
- Set **Human pacing** to **Off** for faster operation.

Click **Start solver**. The app detects the visible question set automatically and skips tabs already marked correct; you do not enter a question count. Existing correct answers have no saved workings or bookwork answers unless they were previously recorded by the app.

The panel stays on top and greys out while working. Press **Esc** or **Stop** to cancel. If hidden, press **F8** to restore it. Read the final status: a stopped run or a question needing review does not mean the homework is complete.

Use **Pause** to correct an answer or inspect a stuck dropdown, then **Resume** to continue the same run. Pending grading, verified answers, and workings are retained. Repeating control sequences trigger bounded replanning, then a logged skip to a verified unfinished question; if navigation is unsafe, the solver pauses for manual repair. A decision received during a pause is discarded. After an application restart, open the same saved set; correct browser tabs are skipped and saved answers remain available. Uncertain submissions still need review.

After the first confirmed wrong submission, reasoning increases from low to high. After four failures, the app records **Needs Human intervention** and attempts to move to the next question. Review these questions yourself.

## 5. Bookwork checks and workings

When enabled, the solver attempts the final bookwork check using verified answers saved in the current homework record. Supported matching covers numeric answers and numeric answer-image cards. Missing, ambiguous, unreadable, or unsupported answers require human review. Successful completion includes verifying **Passed** and clicking Continue.

The **Workings** page displays formatted equations. Choose **Save Markdown** to export verified workings, **Copy workings** to copy LaTeX source, or **Open folder** to find automatic checkpoints. Markdown exports display equations as images: keep the accompanying image-assets folder beside the `.md` file when moving or sharing it. A standalone `.html` copy is also saved with embedded images; open it in a browser to read or print the workings without a LaTeX plugin or internet connection. **Open preview** opens this browser copy directly.

Only captured questions with supported, verified workings appear in the export. Check the coverage and review list before copying calculations into your book.

## 6. Settings and troubleshooting

| Need | Where to look |
| --- | --- |
| Faster entry | **Configuration → Automation:** enable **Numeric keyboard entry** and **Fast submit**; disable **Input delays** and **Smooth pointer**. |
| Adjust pacing | **Waiting Time** ranges from 0–5 seconds and applies when input delays are enabled. It does not remove API or page-loading time. |
| Completion notification | Enable **Finish sound** in Configuration; also check the final status. |
| See what is happening | Enable **Show live activity log on Solver** in Configuration. |
| Investigate a stopped run | Read the status below Start, then choose **Configuration → Open session logs**. Stopping automation keeps the panel open; logs also capture unexpected failures. |
| Track API use | Open **Usage** for session input/output token counts reported by the API. These are not a billing statement. |
| Change the appearance | Open **Appearance** to adjust the available visual settings and animations. |
| Connection timeout | Check internet access, the server URL, credentials, and model availability. For a local server, confirm it is running and reachable. |
| Repeated clicks or no progress | Use **Pause**, correct the controls in the browser, then **Resume**. Automatic loop recovery also saves review logs. |
| Question tabs temporarily unreadable | Startup retries for up to eight seconds, then pauses with progress retained. Expose the full question-tab row and Summary, then **Resume**. |
| Uncertain saved submission | Inspect the browser, then use **Review pending**. This discards uncertain outcomes; it does not mark answers correct. |
| Missing workings | Confirm **Save workings** was enabled before starting and inspect the Workings coverage/review list. |
| Question needs help | Use **Configuration → Open human review logs**, then review the question in the browser. |

Homework records, checkpoints, and review logs stay local. Treat exported files and screenshots as private when sharing them.

With **Fast submit** enabled, a completed plain numeric entry is submitted immediately when Submit is enabled, without another full-field scan or model decision. This trusts the final-entry signal and can submit a mistaken entry; Sparx's grading still controls retries, progress, and which answers are saved as correct. Turn it off for full readback before submission. Fractions, expression selectors, and unfamiliar controls retain their normal readback checks. Identical screenshots reuse one readback, and short grading transitions are handled locally.
