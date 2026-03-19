# Homework 1 - Document Enhancement

This folder contains the solution for Homework 1 (preprocessing, global histogram equalization, CLAHE, and final visualization).

## Contents

- `data/dobosi_peter_laszlo.jpg`: Input image.
- `data/description.pdf`: Assignment statement and formal requirements.
- `src/mw79on_submission/main.py`: Full implementation.
- `output/task4_results.png`: Final generated 2x3 comparison figure.
- `RESULTS.md`: Method summary and requirement compliance notes.
- `REPORT.pdf`: Submission-ready report document.

## Requirements

- Python `>=3.10`
- Packages:
	- `opencv-python`
	- `numpy`
	- `matplotlib`

Install dependencies (from this `homework_1/` directory):

```bash
pip install -e .
```

## Reproduce the Results

From `homework_1/`, run:

```bash
python src/mw79on_submission/main.py
```

This executes the full pipeline:

1. Preprocesses the input image (rotation and border reduction).
2. Applies global histogram equalization on HSV V channel with blending.
3. Applies CLAHE on HSV V channel.
4. Computes RGB histograms for all three outputs.
5. Produces a single `2x3` subplot figure and saves it to:
	 - `output/task4_results.png`

Notes:

- The script currently displays intermediate images and the final figure on screen.
- If you run in a headless environment, use a GUI-capable session or adapt plotting backend/display calls.

Build the PDF report (from `homework_1/`):

```bash
pdflatex -interaction=nonstopmode -halt-on-error REPORT.tex
```

## Formal Requirement Coverage

- Input image uses the my own named file from `data/`.
- OpenCV, NumPy, and Matplotlib are used.
- The code is structured into separate functions for loading/preprocessing, enhancements, histogram calculation, and plotting.
- The final figure contains titles and RGB histograms with intensity axis scaled to `0-255`.

## Notes on Documentation Consistency

- Some course references use terms/examples that can look inconsistent across OpenCV language variants.
- This implementation is validated against Python OpenCV APIs (`cv2.cvtColor`, `cv2.equalizeHist`, `cv2.createCLAHE`, `cv2.calcHist`) and keeps HSV V-channel processing explicit.
- For feature-based methods introduced later, see `homework_2/README.md` where descriptor/matcher compatibility is documented.

## Personal Notes  
  
- It was a fun bite-sized assignment, I've completed it in a couple hours, with this being my first time using opencv, especially with python.
- My initial impressions of the official opencv docs were somewhat mixed. Coming from other native Python libraries this experience could use a bit more refinement, but it is workable. I guess in professional contexts this library is mainly used with other languages.  
- The lab notes/slides were slightly incorrect in some places, I suspect mainly stemming from the author using the library with other languages in their day to day activites?  
- _FYI: While I haven't personally tried this approach, I suspect that a modern agentic LLM in an appropriate harness could probably 'one-shot' this assignment as is._  
  