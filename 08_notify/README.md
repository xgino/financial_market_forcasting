# 08_notify: the monthly email

`email_report.py` mails you only when it matters: for the assets in `settings.NOTIFY["assets"]` (default gold),
only on the calls in `settings.NOTIFY["only_when"]` (default BUY). The mail explains itself: what to do, why
(the votes, the trusted model and its track record, what drives it), what could happen to your money, how
this rule did in the past, and how fresh the data is.

Setup once: copy `secrets.example.env` to `secrets.env`, fill in a Gmail app password, add `secrets.env` to
`.gitignore`. Every month `tools/monthly_update.sh` runs everything including this step.
Preview without sending: `data/notify/last_email.html` (also on the dashboard page Email).