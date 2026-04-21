---
name: TranscribePNGFolder
description: Use this skill whenever the user wants to transcribe PDF contents, contained in PNG files in a local folder, into text or markdown. The source PDFs include academic papers, reports, manuals, or any PDF with mixed text and figures. The skill iterates over the PNG files, uses a vision model to transcribe the text and describe any figures in place, collating everything into a single clean markdown document. Triggers: "transcribe this PDF in directory", "extract text from PDF", "convert PDF contained in directory to markdown", "summarise this PDF" (as a first step).
---

# TranscribePNGFolder Skill

## Overview

Transcribes PDF contents from PNG to a markdown document by:
1. Iterating over PNG files in a given directory
2. Sending each PNG to a vision-capable model and transcribing the contents, including images
3. Sanity checking the output to make sure all figures are present and figures are positioned inline.
4. Write the markdown content to file. 

Figures, charts, and diagrams are described in-line at the position they appear on the page. Do not bunch figure descriptions at the end of the markdown description. Page numbers and boilerplate footers are omitted.

---

## Step 1 — Iterate over PDF contents contained in PNG files

The user will have provided a directory that contains the contents of a PDF, exported to PNG files with pdftoppm. These are named page-01.png, page-02.png, page-03.png, etc...

Once 20 pages have been loaded, call the compress tool to manage context, using the last message as the endId and the message where page processing began as the startId. Write an intermediate markdown file with the transcribed content if necessary, so that it can be edited to add the following pages. 


---

## Step 2 — Transcribe Each Page (vision model)

Place content from all pages into a single markdown document, in the present working directory, with the same name as the directory containing the PDF contents (plus the .md filetype). 

Rules:
- Transcribe ALL body text and figures faithfully, preserving headings, bullet points, and numbered lists.
- For any figure, chart, diagram, table, or image: insert a markdown block at the position
  where it appears on the page, including your interpretation followed by the actual caption, using this format:

    > **Figure [figure number]**
    > 
    > Interpretation: [ Three to ten sentences describing what the figure shows in all its panels —
    > the types (bar chart, diagram, photograph, etc.), what the axes or labels represent,
    > the key data or relationships it communicates. ]
    > 
    > Caption: [ The transcribed caption text. ]

- Retain the figure locations in the text, so that they are located near the same text as they are in the original PDF.
- Omit page numbers, running headers, and boilerplate footers (e.g. "© 2024 Acme Corp",
  "Confidential", journal/conference name repeated at the top or bottom of every page).
- Output only the transcribed markdown — no preamble, no closing remarks.

---

## Step 3 - Sanity checking.

- Check that each Figure has been summarised within the markdown document. If not, amend the transcribed markdown to include the missing figure summary in the correct place in the text.
- Check that the figure descriptions aren't placed in a bunch at the bottom of the markdown document. They should be in the same location in the markdown as they are in the PDF.  

---

## Step 4 - Write to file.

Write the markdown content to a markdown file in the present working directory, named similarly to the input directory. e.g. input directory is martin1989, output file is martin1989.md. 