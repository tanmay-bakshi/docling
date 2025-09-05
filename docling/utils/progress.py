from __future__ import annotations

import threading
from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple


class ProgressReporter(ABC):
    """Interface for reporting conversion progress.

    Implementations can display progress bars, log updates, or no-op.

    :param total_docs: The total number of documents in the run.
    :param file: The path of the document currently being processed.
    :param total_pages: The total number of pages for the document, if applicable.
    :param stage: The pipeline stage identifier (e.g., "Build", "Assemble", "Enrich").
    :param total: The total units for the stage (e.g., number of pages), if applicable.
    :param advance: The number of units to advance the current stage by.
    """

    @abstractmethod
    def start_run(self, total_docs: int) -> None:
        """Mark the beginning of a run.

        :param total_docs: Total number of documents to process.
        :returns: None
        """

    @abstractmethod
    def end_run(self) -> None:
        """Mark the end of a run.

        :returns: None
        """

    @abstractmethod
    def start_document(self, file: Path, total_pages: Optional[int]) -> None:
        """Mark the beginning of processing a document.

        :param file: Path to the input document.
        :param total_pages: Total pages in document, or None if not paginated.
        :returns: None
        """

    @abstractmethod
    def end_document(self, file: Path) -> None:
        """Mark the end of processing a document.

        :param file: Path to the input document.
        :returns: None
        """

    @abstractmethod
    def advance_documents(self, advance: int = 1) -> None:
        """Advance the overall documents progress.

        :param advance: Number of documents to advance by.
        :returns: None
        """

    @abstractmethod
    def start_stage(
        self, file: Path, stage: str, total: Optional[int]
    ) -> None:
        """Mark the beginning of a pipeline stage for a document.

        :param file: Path to the input document.
        :param stage: Stage name (e.g., "Build").
        :param total: Total units for the stage, or None if unknown.
        :returns: None
        """

    @abstractmethod
    def advance_stage(self, file: Path, stage: str, advance: int = 1) -> None:
        """Advance a specific stage for a document.

        :param file: Path to the input document.
        :param stage: Stage name.
        :param advance: Units to advance.
        :returns: None
        """

    @abstractmethod
    def end_stage(self, file: Path, stage: str) -> None:
        """Mark the end of a pipeline stage for a document.

        :param file: Path to the input document.
        :param stage: Stage name.
        :returns: None
        """


class NullProgressReporter(ProgressReporter):
    """No-op progress reporter.

    Useful as a default when progress reporting is disabled.
    """

    def start_run(self, total_docs: int) -> None:
        return None

    def end_run(self) -> None:
        return None

    def start_document(self, file: Path, total_pages: Optional[int]) -> None:
        return None

    def end_document(self, file: Path) -> None:
        return None

    def advance_documents(self, advance: int = 1) -> None:
        return None

    def start_stage(self, file: Path, stage: str, total: Optional[int]) -> None:
        return None

    def advance_stage(self, file: Path, stage: str, advance: int = 1) -> None:
        return None

    def end_stage(self, file: Path, stage: str) -> None:
        return None


class RichProgressReporter(ProgressReporter):
    """Rich-based progress reporter with multiple tasks.

    Uses Rich's progress bars for a run-wide documents task and per-document stage tasks.
    """

    def __init__(self) -> None:
        self._log = logging.getLogger(__name__)
        # Lazy import to avoid hard dependency for non-CLI contexts
        from rich.progress import (
            BarColumn,
            Progress,
            TaskID,
            TaskProgressColumn,
            TextColumn,
            TimeRemainingColumn,
        )

        self._Progress = Progress  # type: ignore[assignment]
        self._TextColumn = TextColumn  # type: ignore[assignment]
        self._BarColumn = BarColumn  # type: ignore[assignment]
        self._TaskProgressColumn = TaskProgressColumn  # type: ignore[assignment]
        self._TimeRemainingColumn = TimeRemainingColumn  # type: ignore[assignment]

        self._progress = Progress(
            self._TextColumn("{task.description}"),
            self._BarColumn(bar_width=None),
            self._TaskProgressColumn(),
            self._TimeRemainingColumn(),
            expand=True,
        )
        self._run_task: Optional["TaskID"] = None
        self._stage_tasks: Dict[Tuple[str, str], "TaskID"] = {}
        self._lock = threading.Lock()
        self._started = False

    def _ensure_started(self) -> None:
        if not self._started:
            self._progress.start()
            self._started = True

    def start_run(self, total_docs: int) -> None:
        self._ensure_started()
        with self._lock:
            if self._run_task is None:
                # Ensure non-negative total.
                total = total_docs if total_docs >= 0 else 0
                self._run_task = self._progress.add_task(
                    "Documents", total=total
                )

    def end_run(self) -> None:
        with self._lock:
            if self._run_task is not None:
                # Mark documents task complete if not already.
                if self._progress.tasks[self._run_task].finished is False:
                    total = self._progress.tasks[self._run_task].total
                    if total is not None:
                        self._progress.update(self._run_task, completed=total)
            # Stop rendering
            if self._started:
                self._progress.stop()
                self._started = False

    def start_document(self, file: Path, total_pages: Optional[int]) -> None:
        # No separate document-level task beyond stages; nothing to do here.
        return None

    def end_document(self, file: Path) -> None:
        # Remove stage tasks for this file to keep UI clean.
        with self._lock:
            keys_to_remove = [key for key in self._stage_tasks if key[0] == str(file)]
            for key in keys_to_remove:
                task_id = self._stage_tasks.pop(key)
                try:
                    self._progress.remove_task(task_id)
                except Exception:
                    # If already removed or invalid, log at debug level.
                    self._log.debug("Failed to remove stage task", exc_info=True)

    def advance_documents(self, advance: int = 1) -> None:
        with self._lock:
            if self._run_task is not None:
                self._progress.update(self._run_task, advance=advance)

    def start_stage(
        self, file: Path, stage: str, total: Optional[int]
    ) -> None:
        self._ensure_started()
        with self._lock:
            key = (str(file), stage)
            if key not in self._stage_tasks:
                desc = f"{file.name} • {stage}"
                # Rich supports total=None for indeterminate tasks
                self._stage_tasks[key] = self._progress.add_task(
                    desc, total=total
                )

    def advance_stage(self, file: Path, stage: str, advance: int = 1) -> None:
        with self._lock:
            key = (str(file), stage)
            task_id = self._stage_tasks.get(key)
            if task_id is not None:
                self._progress.update(task_id, advance=advance)

    def end_stage(self, file: Path, stage: str) -> None:
        with self._lock:
            key = (str(file), stage)
            task_id = self._stage_tasks.get(key)
            if task_id is not None:
                task = self._progress.tasks[task_id]
                # If the task has a finite total, mark as complete; otherwise stop it.
                if task.total is not None:
                    self._progress.update(task_id, completed=task.total)
                try:
                    self._progress.stop_task(task_id)
                except Exception:
                    # If stopping fails (e.g., already stopped), log at debug level.
                    self._log.debug("Failed to stop stage task", exc_info=True)
