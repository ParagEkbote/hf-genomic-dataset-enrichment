from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx2

NCBI_BASE_URL = "https://api.ncbi.nlm.nih.gov/datasets/v2"

DEFAULT_TIMEOUT = httpx2.Timeout(
    connect=10.0,
    read=30.0,
    write=30.0,
    pool=10.0,
)

# Retry policy for transient HTTP failures.
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = (0.0, 0.5, 1.0)

# Sanity bound for record_id input coming from Marimo UI fields.
MAX_RECORD_ID_LENGTH = 100


# ============================================================================
# Exceptions
# ============================================================================


class NCBIError(RuntimeError):
    """Base exception for NCBI client errors."""


class NCBINotFoundError(NCBIError):
    """Raised when NCBI does not return the requested resource."""


class NCBIRequestError(NCBIError):
    """Raised when an NCBI request fails."""


class NCBIResponseError(NCBIError):
    """Raised when an NCBI response has an unexpected schema."""


def _validate_record_id(record_id: str) -> str:
    """
    Reject obviously invalid record_id input early.

    No accession regex is enforced deliberately: the dataset may
    eventually contain identifiers other than NC_..., so this only
    guards against empty or unreasonably long input.
    """

    record_id = record_id.strip()

    if not record_id:
        raise ValueError("record_id must not be empty.")

    if len(record_id) > MAX_RECORD_ID_LENGTH:
        raise ValueError("record_id is unexpectedly long.")

    return record_id


# ============================================================================
# Data models
# ============================================================================


@dataclass(frozen=True)
class SequenceResolution:
    """Result of resolving a nucleotide accession to assembly accessions."""

    accession: str
    assembly_accessions: tuple[str, ...]
    raw: dict[str, Any]


@dataclass(frozen=True)
class AssemblyInfo:
    """Relevant information extracted from an NCBI assembly report."""

    accession: str
    assembly_name: str | None
    tax_id: int | None
    organism_name: str | None
    common_name: str | None
    assembly_level: str | None
    assembly_status: str | None
    raw: dict[str, Any]


@dataclass(frozen=True)
class Taxon:
    """Relevant information extracted from an NCBI taxonomy report."""

    tax_id: int
    scientific_name: str
    common_name: str | None
    rank: str | None
    lineage: tuple[str, ...]
    classification: dict[str, int | str]
    raw: dict[str, Any]


@dataclass(frozen=True)
class TaxonImageMetadata:
    """Metadata describing an NCBI taxonomy image."""

    tax_id: int
    src: str
    license: str | None
    attribution: str | None
    source: str | None
    image_sizes: tuple[str, ...]
    format: str | None
    license_url: str | None


@dataclass(frozen=True)
class SpecimenNCBI:
    """
    Complete NCBI enrichment for a dataset record.

    record_id remains the primary identity.
    NCBI accession, assembly accession, and TaxID are enrichment fields.
    """

    record_id: str
    sequence: SequenceResolution
    assembly: AssemblyInfo

    tax_id: int
    scientific_name: str
    common_name: str | None
    rank: str | None

    lineage: tuple[str, ...]
    classification: dict[str, int | str]

    image: bytes | None = None
    image_metadata: TaxonImageMetadata | None = None
    links: dict[str, Any] | None = None


# ============================================================================
# NCBI client
# ============================================================================


class NCBIClient:
    """Small client for the NCBI Datasets v2 REST API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout: httpx2.Timeout | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("NCBI_API_KEY")
        self.timeout = timeout or DEFAULT_TIMEOUT

        headers = {
            "Accept": "application/json",
            "User-Agent": "carbon-enrichment/1.0",
        }

        if self.api_key:
            headers["api-key"] = self.api_key

        self._client = httpx2.Client(
            base_url=NCBI_BASE_URL,
            headers=headers,
            timeout=self.timeout,
            follow_redirects=True,
        )

    def __enter__(self) -> NCBIClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    # ------------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------------

    def _request_with_retry(self, path: str) -> httpx2.Response:
        """
        Perform a GET request with a small bounded retry for transient
        failures (429 / 500 / 502 / 503 / 504).

        4xx errors other than the retryable ones (e.g. 400, 404) are
        never retried.
        """

        last_exc: httpx2.HTTPError | None = None

        for attempt, delay in enumerate(RETRY_BACKOFF_SECONDS):
            if delay:
                time.sleep(delay)

            try:
                response = self._client.get(path)

            except httpx2.HTTPError as exc:
                last_exc = exc
                if attempt == len(RETRY_BACKOFF_SECONDS) - 1:
                    raise NCBIRequestError(
                        f"NCBI request failed for {path}: {exc}"
                    ) from exc
                continue

            if response.status_code in RETRYABLE_STATUS_CODES:
                if attempt == len(RETRY_BACKOFF_SECONDS) - 1:
                    raise NCBIRequestError(
                        f"NCBI returned HTTP {response.status_code} for "
                        f"{path} after {MAX_RETRIES} attempts: "
                        f"{response.text[:500]}"
                    )
                continue

            return response

        # Unreachable in practice, but keeps type-checkers satisfied.
        raise NCBIRequestError(f"NCBI request failed for {path}: {last_exc}")

    def _get(self, path: str) -> dict[str, Any]:
        """Perform a GET request and return a JSON object."""

        response = self._request_with_retry(path)

        if response.status_code == 404:
            raise NCBINotFoundError(f"NCBI resource not found: {path}")

        try:
            response.raise_for_status()

        except httpx2.HTTPStatusError as exc:
            raise NCBIRequestError(
                f"NCBI returned HTTP {response.status_code} for {path}: "
                f"{response.text[:500]}"
            ) from exc

        try:
            data = response.json()

        except ValueError as exc:
            raise NCBIResponseError(f"NCBI returned invalid JSON for {path}") from exc

        if not isinstance(data, dict):
            raise NCBIResponseError(
                f"Unexpected NCBI response type for {path}: {type(data).__name__}"
            )

        return data

    def _get_bytes(self, path: str) -> bytes:
        """Perform a GET request and return the raw response bytes."""

        response = self._request_with_retry(path)

        if response.status_code == 404:
            raise NCBINotFoundError(f"NCBI resource not found: {path}")

        try:
            response.raise_for_status()

        except httpx2.HTTPStatusError as exc:
            raise NCBIRequestError(
                f"NCBI returned HTTP {response.status_code} for {path}: "
                f"{response.text[:500]}"
            ) from exc

        return response.content

    # =========================================================================
    # Sequence accession -> assembly
    # =========================================================================

    def resolve_sequence_accession(
        self,
        accession: str,
    ) -> SequenceResolution:
        """
        Resolve a nucleotide sequence accession to assembly accessions.

        Example:
            NC_088780.1
                ->
            GCF_958450345.1
        """

        path = f"/genome/sequence_accession/{accession}/sequence_assemblies"

        data = self._get(path)

        accessions = data.get("accessions", [])

        if not isinstance(accessions, list):
            raise NCBIResponseError(
                "Unexpected sequence-resolution response: 'accessions' is not a list."
            )

        assembly_accessions = tuple(
            value
            for value in accessions
            if isinstance(value, str)
            and (value.startswith("GCA_") or value.startswith("GCF_"))
        )

        return SequenceResolution(
            accession=accession,
            assembly_accessions=assembly_accessions,
            raw=data,
        )

    # =========================================================================
    # Assembly
    # =========================================================================

    def get_assembly_report(
        self,
        accession: str,
    ) -> dict[str, Any]:
        """Return the raw NCBI assembly dataset report."""

        path = f"/genome/accession/{accession}/dataset_report"

        data = self._get(path)

        reports = data.get("reports")

        if not isinstance(reports, list):
            raise NCBIResponseError(
                "Unexpected assembly response: 'reports' is not a list."
            )

        return data

    def get_assembly(
        self,
        accession: str,
    ) -> AssemblyInfo:
        """Parse an NCBI assembly dataset report."""

        data = self.get_assembly_report(accession)

        reports = data["reports"]

        if not reports:
            raise NCBINotFoundError(f"No assembly report returned for {accession}")

        report = reports[0]

        if not isinstance(report, dict):
            raise NCBIResponseError("Unexpected assembly report entry.")

        organism = report.get("organism") or {}
        assembly_info = report.get("assembly_info") or {}

        if not isinstance(organism, dict):
            organism = {}

        if not isinstance(assembly_info, dict):
            assembly_info = {}

        tax_id = organism.get("tax_id")

        return AssemblyInfo(
            accession=report.get(
                "current_accession",
                report.get("accession", accession),
            ),
            assembly_name=assembly_info.get("assembly_name"),
            tax_id=tax_id if isinstance(tax_id, int) else None,
            organism_name=organism.get("organism_name"),
            common_name=organism.get("common_name"),
            assembly_level=assembly_info.get("assembly_level"),
            assembly_status=assembly_info.get("assembly_status"),
            raw=report,
        )

    # =========================================================================
    # Taxonomy
    # =========================================================================

    def get_taxon_report(
        self,
        taxon: int | str,
    ) -> dict[str, Any]:
        """Return the raw NCBI taxonomy dataset report."""

        path = f"/taxonomy/taxon/{taxon}/dataset_report"

        data = self._get(path)

        reports = data.get("reports")

        if not isinstance(reports, list):
            raise NCBIResponseError(
                "Unexpected taxonomy response: 'reports' is not a list."
            )

        return data

    def get_taxon(
        self,
        taxon: int | str,
    ) -> Taxon:
        """
        Parse an NCBI taxonomy dataset report.

        Current NCBI response structure:

            reports[0].taxonomy
                tax_id
                rank
                current_scientific_name.name
                curator_common_name
                classification
        """

        data = self.get_taxon_report(taxon)

        reports = data["reports"]

        if not reports:
            raise NCBINotFoundError(f"No taxonomy report returned for {taxon}")

        report = reports[0]

        if not isinstance(report, dict):
            raise NCBIResponseError("Unexpected taxonomy report entry.")

        taxonomy = report.get("taxonomy")

        if not isinstance(taxonomy, dict):
            raise NCBIResponseError(
                "Unexpected taxonomy response: "
                "'taxonomy' is missing or is not an object."
            )

        # ---------------------------------------------------------------------
        # TaxID
        # ---------------------------------------------------------------------

        tax_id = taxonomy.get("tax_id")

        if not isinstance(tax_id, int):
            raise NCBIResponseError(
                "Unexpected taxonomy response: 'taxonomy.tax_id' is missing or invalid."
            )

        # ---------------------------------------------------------------------
        # Scientific name
        # ---------------------------------------------------------------------

        scientific_name_data = taxonomy.get("current_scientific_name")

        if not isinstance(scientific_name_data, dict):
            raise NCBIResponseError(
                "Unexpected taxonomy response: "
                "'current_scientific_name' is missing or invalid."
            )

        scientific_name = scientific_name_data.get("name")

        if not isinstance(scientific_name, str):
            raise NCBIResponseError(
                "Unexpected taxonomy response: "
                "'current_scientific_name.name' is missing."
            )

        # ---------------------------------------------------------------------
        # Common name
        # ---------------------------------------------------------------------

        common_name = taxonomy.get("curator_common_name")

        if not isinstance(common_name, str):
            common_name = None

        # ---------------------------------------------------------------------
        # Rank
        # ---------------------------------------------------------------------

        rank = taxonomy.get("rank")

        if not isinstance(rank, str):
            rank = None

        # ---------------------------------------------------------------------
        # Classification
        # ---------------------------------------------------------------------

        classification_raw = taxonomy.get("classification") or {}

        if not isinstance(classification_raw, dict):
            raise NCBIResponseError(
                "Unexpected taxonomy response: 'classification' is not an object."
            )

        classification: dict[str, int | str] = {}
        lineage: list[str] = []

        # NCBI's classification is keyed by taxonomic rank.
        #
        # We preserve the biologically useful top-to-bottom ordering
        # rather than relying on JSON object insertion order.
        rank_order = (
            "domain",
            "superkingdom",
            "kingdom",
            "subkingdom",
            "phylum",
            "subphylum",
            "class",
            "subclass",
            "order",
            "suborder",
            "family",
            "subfamily",
            "genus",
            "subgenus",
            "species",
            "subspecies",
        )

        for rank_name in rank_order:
            node = classification_raw.get(rank_name)

            if not isinstance(node, dict):
                continue

            name = node.get("name")
            node_id = node.get("id")

            if isinstance(name, str):
                lineage.append(name)
                classification[f"{rank_name}_name"] = name

            if isinstance(node_id, int):
                classification[f"{rank_name}_tax_id"] = node_id

        return Taxon(
            tax_id=tax_id,
            scientific_name=scientific_name,
            common_name=common_name,
            rank=rank,
            lineage=tuple(lineage),
            classification=classification,
            raw=data,
        )

    # =========================================================================
    # Taxonomy search / enrichment
    # =========================================================================

    def search_taxa(
        self,
        query: str,
    ) -> dict[str, Any]:
        """
        Search NCBI taxonomy names.

        Uses the NCBI taxonomy suggestion endpoint. The query is
        user-entered from Marimo, so it is URL-encoded before being
        interpolated into the path.
        """

        query_encoded = quote(query, safe="")
        path = f"/taxonomy/taxon_suggest/{query_encoded}"

        return self._get(path)

    def get_taxon_image(
        self,
        taxon: int | str,
    ) -> bytes:
        """Return the raw NCBI taxonomy image bytes."""

        path = f"/taxonomy/taxon/{taxon}/image"

        return self._get_bytes(path)

    def get_taxon_image_metadata(
        self,
        taxon: int | str,
    ) -> TaxonImageMetadata:
        """Return parsed metadata for the NCBI taxonomy image."""

        path = f"/taxonomy/taxon/{taxon}/image/metadata"

        data = self._get(path)

        tax_id_raw = data.get("tax_id")

        try:
            tax_id = int(tax_id_raw)
        except (TypeError, ValueError) as exc:
            raise NCBIResponseError(
                "Unexpected image metadata response: 'tax_id' is missing or invalid."
            ) from exc

        image_sizes_raw = data.get("image_sizes", [])

        if not isinstance(image_sizes_raw, list):
            image_sizes_raw = []

        image_sizes = tuple(
            value for value in image_sizes_raw if isinstance(value, str)
        )

        return TaxonImageMetadata(
            tax_id=tax_id,
            src=data.get("src", ""),
            license=data.get("license"),
            attribution=data.get("attribution"),
            source=data.get("source"),
            image_sizes=image_sizes,
            format=data.get("format"),
            license_url=data.get("license_url"),
        )

    def get_taxon_links(
        self,
        taxon: int | str,
    ) -> dict[str, Any]:
        """Return NCBI links associated with a taxon."""

        path = f"/taxonomy/taxon/{taxon}/links"

        return self._get(path)

    def get_taxon_media(
        self,
        taxon: int | str,
        *,
        include_image: bool = False,
    ) -> tuple[bytes | None, TaxonImageMetadata, dict[str, Any]]:
        """
        Retrieve taxonomy image metadata, links, and optionally image bytes.

        Image bytes are not downloaded unless include_image=True.
        """

        metadata = self.get_taxon_image_metadata(taxon)
        links = self.get_taxon_links(taxon)

        image = None

        if include_image:
            image = self.get_taxon_image(taxon)

        return image, metadata, links

    # =========================================================================
    # Complete record resolution
    # =========================================================================

    def resolve_record(
        self,
        record_id: str,
    ) -> SpecimenNCBI:
        """
        Resolve a dataset record through NCBI.

        Resolution chain:

            record_id
                ↓
            sequence accession
                ↓
            assembly accession
                ↓
            assembly report
                ↓
            TaxID
                ↓
            taxonomy report
                ↓
            specimen NCBI context

        Note: this intentionally does not fetch image bytes, image
        metadata, or links. Those are attached lazily via
        get_taxon_media() once the UI actually needs them, to keep
        this call cheap in a reactive Marimo notebook.

        Each stage is wrapped so the caller (and the Marimo UI) can
        tell which stage failed: sequence resolution, assembly
        lookup, or taxonomy lookup.
        """

        record_id = _validate_record_id(record_id)

        # ---------------------------------------------------------------------
        # 1. Sequence accession -> assembly accession(s)
        # ---------------------------------------------------------------------

        try:
            sequence = self.resolve_sequence_accession(record_id)

        except NCBIError as exc:
            raise NCBIRequestError(
                f"Failed resolving sequence accession for record {record_id}: {exc}"
            ) from exc

        if not sequence.assembly_accessions:
            raise NCBINotFoundError(
                f"No assembly accession found for sequence accession {record_id}"
            )

        # Current policy: use the first NCBI-resolved assembly when
        # more than one is returned. The complete set remains
        # available through sequence.assembly_accessions, so a caller
        # that needs a different assembly can resolve it explicitly
        # via get_assembly() using one of those accessions.
        assembly_accession = sequence.assembly_accessions[0]

        # ---------------------------------------------------------------------
        # 2. Assembly -> organism / TaxID
        # ---------------------------------------------------------------------

        try:
            assembly = self.get_assembly(assembly_accession)

        except NCBIError as exc:
            raise NCBIRequestError(
                f"Failed resolving assembly {assembly_accession} for "
                f"record {record_id}: {exc}"
            ) from exc

        if assembly.tax_id is None:
            raise NCBIResponseError(
                f"Assembly {assembly_accession} has no organism tax_id."
            )

        # ---------------------------------------------------------------------
        # 3. TaxID -> taxonomy
        # ---------------------------------------------------------------------

        try:
            taxon = self.get_taxon(assembly.tax_id)

        except NCBIError as exc:
            raise NCBIRequestError(
                f"Failed resolving taxonomy for record {record_id}: {exc}"
            ) from exc

        # ---------------------------------------------------------------------
        # 4. Combine into specimen context
        # ---------------------------------------------------------------------

        return SpecimenNCBI(
            record_id=record_id,
            sequence=sequence,
            assembly=assembly,
            tax_id=taxon.tax_id,
            scientific_name=taxon.scientific_name,
            common_name=taxon.common_name,
            rank=taxon.rank,
            lineage=taxon.lineage,
            classification=taxon.classification,
        )


# ============================================================================
# One-shot convenience function
# ============================================================================


def resolve_record(
    record_id: str,
    *,
    api_key: str | None = None,
) -> SpecimenNCBI:
    """
    Resolve one record without explicitly creating an NCBIClient.

    For a persistent Marimo notebook, prefer:

        ncbi = NCBIClient()

    and reuse that instance.
    """

    with NCBIClient(api_key=api_key) as ncbi:
        return ncbi.resolve_record(record_id)


# ============================================================================
# Public API
# ============================================================================


__all__ = [
    "NCBIClient",
    "NCBIError",
    "NCBINotFoundError",
    "NCBIRequestError",
    "NCBIResponseError",
    "SequenceResolution",
    "AssemblyInfo",
    "Taxon",
    "TaxonImageMetadata",
    "SpecimenNCBI",
    "resolve_record",
]
