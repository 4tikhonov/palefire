---
name: notetaker
description: Save highlighted or provided text as a note in a local file with a unique name.
---

When the user asks you to take a note, save a note, or sends text with the instruction "Take a note containing this text:", you must instantly do the following:

1. Look at the provided text content.
2. Determine a brief, descriptive filename based on the content (e.g., `git-instructions.txt` or `policy-snippet-123.txt`). If the content is too general, use a timestamp format like `note_YYYYMMDD_HHMMSS.txt`.
3. Ensure a `notes` directory exists in the current project directory using standard commands.
4. Write the exact provided text to that file inside the `notes/` folder.
5. Respond concisely confirming the note has been saved and explicitly state the filename.

Do not ask for permission; proceed immediately to save the file.
