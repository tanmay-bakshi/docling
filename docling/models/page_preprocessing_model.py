import re
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, Optional
import logging
import traceback

import numpy as np
from PIL import ImageDraw
from pydantic import BaseModel

from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.settings import settings
from docling.models.base_model import BasePageModel
from docling.utils.profiling import TimeRecorder
from docling.utils.progress import NullProgressReporter, ProgressReporter


class PagePreprocessingOptions(BaseModel):
    images_scale: Optional[float]
    skip_cell_extraction: bool = (
        False  # Skip text cell extraction for VLM-only processing
    )


class PagePreprocessingModel(BasePageModel):
    def __init__(self, options: PagePreprocessingOptions):
        self.options = options
        self._progress_reporter: ProgressReporter = NullProgressReporter()
        self._progress_prefix: str = ""
        self._progress_file: Optional[Path] = None
        self._log = logging.getLogger(__name__)

        # Pre-compiled regex patterns for efficiency
        self.GLYPH_RE = re.compile(r"GLYPH<[0-9A-Fa-f]+>")
        self.SLASH_G_RE = re.compile(r"(?:/G\d+){2,}")
        self.FRAG_RE = re.compile(r"\b[A-Za-z](?:/[a-z]{1,3}\.[a-z]{1,3}){2,}\b")
        self.SLASH_NUMBER_GARBAGE_RE = re.compile(
            r"(?:/\w+\s*){2,}"
        )  # Two or more "/token " sequences

    def set_progress_context(
        self, reporter: ProgressReporter, file: Path, batch_ix: int
    ) -> None:
        """Attach a progress reporter and context for hierarchical stages.

        :param reporter: The reporter instance to use.
        :param file: The input document file path.
        :param batch_ix: Zero-based batch index for stage naming.
        :returns: None
        """
        self._progress_reporter = reporter
        self._progress_file = file
        self._progress_prefix = f"Build/Batch{batch_ix + 1}/PagePreprocessingModel"

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
            else:
                with TimeRecorder(conv_res, "page_parse"):
                    page = self._populate_page_images(page)
                    if not self.options.skip_cell_extraction:
                        page = self._parse_page_cells(conv_res, page)
                yield page

    # Generate the page image and store it in the page object
    def _populate_page_images(self, page: Page) -> Page:
        # default scale
        try:
            if self._progress_file is not None:
                self._progress_reporter.start_stage(
                    file=self._progress_file,
                    stage=f"{self._progress_prefix}/Page{page.page_no + 1}/Images/Default",
                    total=1,
                )
            page.get_image(
                scale=1.0
            )  # puts the page image on the image cache at default scale
            if self._progress_file is not None:
                self._progress_reporter.end_stage(
                    file=self._progress_file,
                    stage=f"{self._progress_prefix}/Page{page.page_no + 1}/Images/Default",
                )
        except Exception:
            self._log.debug(
                "Error generating default-scale image.\n%s",
                traceback.format_exc(),
            )

        images_scale = self.options.images_scale
        # user requested scales
        if images_scale is not None:
            try:
                page._default_image_scale = images_scale
                if self._progress_file is not None:
                    self._progress_reporter.start_stage(
                        file=self._progress_file,
                        stage=f"{self._progress_prefix}/Page{page.page_no + 1}/Images/Scaled({images_scale:g})",
                        total=1,
                    )
                page.get_image(
                    scale=images_scale
                )  # this will trigger storing the image in the internal cache
                if self._progress_file is not None:
                    self._progress_reporter.end_stage(
                        file=self._progress_file,
                        stage=f"{self._progress_prefix}/Page{page.page_no + 1}/Images/Scaled({images_scale:g})",
                    )
            except Exception:
                self._log.debug(
                    "Error generating scaled image.\n%s", traceback.format_exc()
                )

        return page

    # Extract and populate the page cells and store it in the page object
    def _parse_page_cells(self, conv_res: ConversionResult, page: Page) -> Page:
        assert page._backend is not None
        # Segmentation
        if self._progress_file is not None:
            self._progress_reporter.start_stage(
                file=self._progress_file,
                stage=f"{self._progress_prefix}/Page{page.page_no + 1}/ParseCells/Segment",
                total=1,
            )
        page.parsed_page = page._backend.get_segmented_page()
        if self._progress_file is not None:
            self._progress_reporter.end_stage(
                file=self._progress_file,
                stage=f"{self._progress_prefix}/Page{page.page_no + 1}/ParseCells/Segment",
            )
        assert page.parsed_page is not None

        # Rate the text quality from the PDF parser, and aggregate on page
        text_scores = []
        # Quality scoring
        total_cells = len(page.cells)
        if self._progress_file is not None:
            self._progress_reporter.start_stage(
                file=self._progress_file,
                stage=f"{self._progress_prefix}/Page{page.page_no + 1}/ParseCells/Quality",
                total=total_cells if total_cells > 0 else None,
            )
        for c in page.cells:
            score = self.rate_text_quality(c.text)
            text_scores.append(score)
            if self._progress_file is not None:
                self._progress_reporter.advance_stage(
                    file=self._progress_file,
                    stage=f"{self._progress_prefix}/Page{page.page_no + 1}/ParseCells/Quality",
                    advance=1,
                )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "Mean of empty slice", RuntimeWarning, "numpy"
            )
            conv_res.confidence.pages[page.page_no].parse_score = float(
                np.nanquantile(
                    text_scores, q=0.10
                )  # To emphasise problems in the parse_score, we take the 10% percentile score of all text cells.
            )
        if self._progress_file is not None:
            self._progress_reporter.end_stage(
                file=self._progress_file,
                stage=f"{self._progress_prefix}/Page{page.page_no + 1}/ParseCells/Quality",
            )

        # DEBUG code:
        def draw_text_boxes(image, cells, show: bool = False):
            draw = ImageDraw.Draw(image.copy())
            stage = f"{self._progress_prefix}/Page{page.page_no + 1}/Debug/DrawTextBoxes"
            if self._progress_file is not None:
                self._progress_reporter.start_stage(
                    file=self._progress_file,
                    stage=stage,
                    total=len(cells) if len(cells) > 0 else None,
                )
            for c in cells:
                x0, y0, x1, y1 = (
                    c.to_bounding_box().l,
                    c.to_bounding_box().t,
                    c.to_bounding_box().r,
                    c.to_bounding_box().b,
                )

                draw.rectangle([(x0, y0), (x1, y1)], outline="red")
                if self._progress_file is not None:
                    self._progress_reporter.advance_stage(
                        file=self._progress_file, stage=stage, advance=1
                    )
            if show:
                image.show()
            else:
                out_path: Path = (
                    Path(settings.debug.debug_output_path)
                    / f"debug_{conv_res.input.file.stem}"
                )
                out_path.mkdir(parents=True, exist_ok=True)

                out_file = out_path / f"cells_page_{page.page_no:05}.png"
                image.save(str(out_file), format="png")
            if self._progress_file is not None:
                self._progress_reporter.end_stage(
                    file=self._progress_file, stage=stage
                )

        if settings.debug.visualize_cells is True:
            draw_text_boxes(page.get_image(scale=1.0), page.cells)

        return page

    def rate_text_quality(self, text: str) -> float:
        # Hard errors: if any of these patterns are found, return 0.0 immediately.
        blacklist_chars = ["�"]
        if (
            any(text.find(c) >= 0 for c in blacklist_chars)
            or self.GLYPH_RE.search(text)
            or self.SLASH_G_RE.search(text)
            or self.SLASH_NUMBER_GARBAGE_RE.match(
                text
            )  # Check if text is mostly slash-number pattern
        ):
            return 0.0

        penalty = 0.0

        # Apply a penalty only if the fragmented words pattern occurs at least three times.
        frag_matches = self.FRAG_RE.findall(text)
        if len(frag_matches) >= 3:
            penalty += 0.1 * len(frag_matches)

        # Additional heuristic: if the average token length is below 2, add a penalty.
        # tokens = text.split()
        # if tokens and (sum(map(len, tokens)) / len(tokens)) < 2:
        #    penalty += 0.2

        return max(1.0 - penalty, 0.0)
