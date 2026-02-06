---
name: screenshot-test
description: Take screenshots of the LEO Viz app and review them visually. Use this skill after making code changes to verify the UI still works correctly, or when you need to see the current state of the application.
---

# Screenshot Test Skill

Capture and review screenshots of the LEO Viz application to verify UI functionality.

## Quick Screenshot (Use This)

Run this single command to start app, capture, and stop immediately:

```bash
pkill -f "target/release/leo-viz" 2>/dev/null
cargo run --release 2>&1 &
sleep 4
osascript -e 'tell application "System Events" to set frontmost of (first process whose name contains "leo-viz") to true' 2>/dev/null
sleep 1
screencapture -x /private/tmp/leo-viz-screenshots/current.png
pkill -f "target/release/leo-viz"
echo "Done"
```

Then read the screenshot with the Read tool:
```
/private/tmp/leo-viz-screenshots/current.png
```

## What to Check

- App window renders without crashes
- 3D Earth/planet view displays correctly
- Left settings panel is visible and readable
- No visual glitches or rendering artifacts
