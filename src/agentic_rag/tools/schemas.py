from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ArxivSortBy(StrEnum):
    relevance = "relevance"
    submitted_date = "submitted_date"
    last_updated_date = "last_updated_date"


class ArxivSortOrder(StrEnum):
    ascending = "ascending"
    descending = "descending"


class ToolStatus(StrEnum):
    ok = "ok"
    partial = "partial"
    error = "error"


class ArxivLookupByIdInput(BaseModel):
    arxiv_ids: list[str] = Field(min_length=1, max_length=50)
    include_abstract: bool = True

    @field_validator("arxiv_ids")
    @classmethod
    def normalize_ids(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value if item.strip()]
        if not normalized:
            raise ValueError("at least one non-empty arxiv id is required")
        return normalized


class ArxivSearchInput(BaseModel):
    query: str = Field(min_length=1)
    categories: list[str] = Field(default_factory=lambda: ["cs.AI"], max_length=10)
    max_results: int = Field(default=5, ge=1, le=10)
    sort_by: ArxivSortBy = ArxivSortBy.relevance
    sort_order: ArxivSortOrder = ArxivSortOrder.descending
    date_from: date | None = None
    date_to: date | None = None

    @field_validator("query")
    @classmethod
    def clean_query(cls, value: str) -> str:
        clean = value.strip()
        if not clean:
            raise ValueError("query must not be empty")
        return clean

    @model_validator(mode="after")
    def validate_dates(self) -> ArxivSearchInput:
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must be earlier than or equal to date_to")
        return self


class ArxivGetRecentInput(BaseModel):
    category: str = "cs.AI"
    days_back: int = Field(default=90, ge=1, le=365)
    max_results: int = Field(default=100, ge=1, le=1000)
    query_filter: str | None = None


class ArxivPaperMetadata(BaseModel):
    arxiv_id: str
    version: str | None = None
    title: str
    authors: list[str]
    abstract: str | None = None
    categories: list[str]
    primary_category: str | None = None
    published_at: datetime | None = None
    updated_at: datetime | None = None
    abs_url: str | None = None
    pdf_url: str | None = None
    doi: str | None = None
    comment: str | None = None


class ArxivToolOutput(BaseModel):
    status: ToolStatus
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    papers: list[ArxivPaperMetadata] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    source: Literal["arxiv_api", "cache"] = "arxiv_api"
