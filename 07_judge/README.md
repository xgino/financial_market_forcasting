# tools

**Monthly reminder** (macOS runs the update and shows a notification with the verdict):
1. Open `nl.paperfund.monthly.plist`, replace `FULL_PATH` with your v2 folder (in Terminal: `pwd` inside v2).
2. `cp tools/nl.paperfund.monthly.plist ~/Library/LaunchAgents/`
3. `launchctl load ~/Library/LaunchAgents/nl.paperfund.monthly.plist`
Your Mac has to be on (or it runs when it wakes up). Test it now: `bash tools/monthly_update.sh`
Stop it: `launchctl unload ~/Library/LaunchAgents/nl.paperfund.monthly.plist`

**Backups**: `bash tools/backup.sh` writes `requirements.lock` and commits `data/live/` to its own git repo.
Once: create a PRIVATE repo on GitHub, then `cd data/live && git remote add origin <its url>`.
After that every backup also pushes. To rebuild everything on another machine: `pip install -r requirements.lock`.