from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


@dataclass
class RevisionEntry:
    date: str
    file: str
    row: int | None
    coded_nonverbal: int | None
    column: str | None
    old_value: object
    new_value: object
    severity: str
    message: str


class RevisionRecorder:
    """
    Collects validation/autocorrection issues during cleaning and writes them
    to an Excel revisions table.

    The table is meant for manual review/correction by coders.
    """

    def __init__(
        self,
        output_path: Path | str,
        date_format: str = "%y%m%d",
    ) -> None:
        self.output_path = Path(output_path)
        self.date_format = date_format
        self.entries: list[RevisionEntry] = []

    def add(
        self,
        *,
        file: Path | str,
        row: int | None,
        column: str | None,
        old_value: object,
        new_value: object = None,
        severity: str,
        message: str,
        coded_nonverbal: int | None = None,
    ) -> None:
        self.entries.append(
            RevisionEntry(
                date=datetime.now().strftime(self.date_format),
                file=str(file),
                row=row,
                coded_nonverbal=coded_nonverbal,
                column=column,
                old_value=self._stringify_value(old_value),
                new_value=self._stringify_value(new_value),
                severity=severity.upper(),
                message=message,
            )
        )

    def to_dataframe(self) -> pd.DataFrame:
        columns = [
            "date",
            "file",
            "row",
            "coded_nonverbal",
            "column",
            "old_value",
            "new_value",
            "severity",
            "message",
        ]

        if not self.entries:
            return pd.DataFrame(columns=columns)

        return pd.DataFrame([asdict(entry) for entry in self.entries], columns=columns)

    def write(self) -> Path:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        df = self.to_dataframe()

        try:
            writer_context = pd.ExcelWriter(self.output_path, engine="openpyxl")
        except PermissionError as exc:
            raise PermissionError(
                f"Could not write revisions workbook at {self.output_path}. "
                "Close the workbook if it is open in Excel, then rerun cleaning."
            ) from exc

        with writer_context as writer:
            df.to_excel(writer, index=False, sheet_name="revisions")

            worksheet = writer.sheets["revisions"]
            worksheet.freeze_panes = "A2"

            widths = {
                "A": 10,  # date
                "B": 60,  # file
                "C": 10,  # row
                "D": 18,  # coded_nonverbal
                "E": 20,  # column
                "F": 22,  # old_value
                "G": 22,  # new_value
                "H": 12,  # severity
                "I": 70,  # message
            }

            for col_letter, width in widths.items():
                worksheet.column_dimensions[col_letter].width = width

            worksheet.auto_filter.ref = worksheet.dimensions

        return self.output_path

    @staticmethod
    def _stringify_value(value: object) -> object:
        """
        Keep real blanks as None in Excel, but stringify weird values safely.
        """
        if value is None:
            return None

        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass

        return str(value)
