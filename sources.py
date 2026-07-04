"""
UAP Source Registry
-------------------
The master catalog of every known UAP/UFO document source on the internet.

Each Source describes WHERE documents live and HOW to enumerate them:
  - static_resources : known, hardcoded download URLs (always included)
  - scraper          : name of a discovery scraper in scrapers.py that
                       enumerates additional per-file URLs at runtime
  - access           : "direct" (static URLs suffice), "scrape" (needs the
                       scraper for full coverage), "manual" (human steps
                       required — documented in notes, listed in the
                       manifest so nothing is silently missing)

Resource dicts produced here and by scrapers share one schema:
  {
    "source":      source id,
    "title":       human title,
    "url":         download URL (or git clone URL for kind=git),
    "kind":        zip|pdf|csv|json|txt|html|image|video|git|torrent,
    "category":    source category,
    "size_hint":   approximate size string or None,
    "sha256":      known checksum or None,
    "requires_ocr": bool — False for text-native data (CSV/JSON/etc),
    "verified":    True if the URL was confirmed working at catalog time,
    "notes":       free text,
  }
"""

from dataclasses import dataclass, field
from typing import Optional


# Categories
US_GOV        = "us_government"
INTL_GOV      = "international_government"
COMMUNITY     = "community_archive"
DATASET       = "structured_dataset"
MIRROR        = "mirror"


def resource(source: str, title: str, url: str, kind: str, category: str,
             size_hint: Optional[str] = None, sha256: Optional[str] = None,
             requires_ocr: bool = True, verified: bool = False,
             notes: str = "") -> dict:
    return {
        "source": source, "title": title, "url": url, "kind": kind,
        "category": category, "size_hint": size_hint, "sha256": sha256,
        "requires_ocr": requires_ocr, "verified": verified, "notes": notes,
    }


@dataclass
class Source:
    id: str
    name: str
    org: str
    homepage: str
    category: str
    description: str
    access: str                      # direct | scrape | manual
    scraper: Optional[str] = None    # key into scrapers.SCRAPERS
    static_resources: list = field(default_factory=list)
    notes: str = ""


# ─────────────────────────────────────────────
# U.S. GOVERNMENT
# ─────────────────────────────────────────────

WARGOV = Source(
    id="wargov_pursue",
    name="Department of War — PURSUE UAP Releases",
    org="U.S. Department of War",
    homepage="https://www.war.gov/ufo/",
    category=US_GOV,
    description=(
        "Presidential Unsealing and Reporting System for UAP Encounters "
        "(PURSUE). Rolling declassified releases starting May 2026: PDFs, "
        "images, and videos spanning 1948-2026 from DoW, FBI, NASA, State."
    ),
    access="scrape",
    scraper="wargov_pursue",
    static_resources=[
        resource("wargov_pursue", "PURSUE Release 01 CSV manifest",
                 "https://www.war.gov/Portals/1/Interactive/2026/UFO/uap-release001.csv",
                 "csv", US_GOV, requires_ocr=False, verified=True,
                 notes="Official per-file catalog for Release 01."),
        resource("wargov_pursue", "PURSUE tracker: all files API (community)",
                 "https://pursueufotracker.com/generated/api/files.json",
                 "json", US_GOV, requires_ocr=False, verified=True,
                 notes="Community JSON API covering all releases; includes war.gov source URLs."),
        resource("wargov_pursue", "PURSUE tracker: verification manifest (SHA-256)",
                 "https://pursueufotracker.com/generated/verification-manifest.json",
                 "json", US_GOV, requires_ocr=False, verified=True,
                 notes="id, title, sha256, size, original war.gov URL for every file."),
    ],
    notes="Individual files follow https://www.war.gov/medialink/ufo/<release>/<filename>.",
)

NARA = Source(
    id="nara_uap",
    name="U.S. National Archives — UAP Records (incl. RG 615 & Project Blue Book)",
    org="NARA",
    homepage="https://www.archives.gov/research/topics/uaps",
    category=US_GOV,
    description=(
        "Bulk ZIP downloads of all digitized UAP records in the National "
        "Archives Catalog: Record Group 615 (UAP Records Collection, 2024 "
        "NDAA transfers from ODNI/OSD/FAA/NRC), Project Blue Book case "
        "files, and related textual/microfilm holdings. ZIPs of images/PDFs "
        "plus JSON metadata per record; refreshed ~3x per year."
    ),
    access="scrape",
    scraper="nara_uap_bulk",
    static_resources=[
        resource("nara_uap", "NARA UAP bulk downloads index page",
                 "https://www.archives.gov/research/catalog/catalog-bulk-downloads/uap-bulk-download",
                 "html", US_GOV, requires_ocr=False, verified=True,
                 notes="Scraper enumerates every ZIP + JSON metadata link on this page."),
    ],
)

CIA = Source(
    id="cia_foia",
    name="CIA FOIA Electronic Reading Room — UFO collections",
    org="CIA",
    homepage="https://www.cia.gov/readingroom/collection/ufos-fact-or-fiction",
    category=US_GOV,
    description=(
        "CIA's declassified UFO holdings (1940s-1990s): the 'UFOs: Fact or "
        "Fiction?' special collection plus CREST records surfaced by keyword "
        "search. ~2,780+ pages as individual PDFs."
    ),
    access="scrape",
    scraper="cia_readingroom",
    static_resources=[],
    notes="The Black Vault CIA CD-ROM ZIP (below) is the fastest bulk path for the same corpus.",
)

FBI = Source(
    id="fbi_vault",
    name="FBI Vault — UFO / Majestic 12 / Roswell / Guy Hottel / Project Blue Book",
    org="FBI",
    homepage="https://vault.fbi.gov/UFO",
    category=US_GOV,
    description=(
        "FBI FOIA reading room. UFO file in 16 parts (1947-1954 era reports, "
        "field office dispatches) plus related files: Majestic 12, Roswell, "
        "the Guy Hottel memo, and Project Blue Book correspondence."
    ),
    access="scrape",
    scraper="fbi_vault",
    static_resources=[],
    notes="PDF pattern: <folder>/<part>/at_download/file on vault.fbi.gov.",
)

NSA = Source(
    id="nsa_ufo",
    name="NSA — Declassified UFO documents",
    org="NSA",
    homepage="https://www.nsa.gov/Helpful-Links/NSA-FOIA/Frequently-Requested-Information/Unidentified-Flying-Objects-UFOs/",
    category=US_GOV,
    description="NSA FOIA 'Frequently Requested' UFO records: SIGINT-related UFO reports, affidavits, and analyses.",
    access="scrape",
    scraper="nsa_ufo",
    static_resources=[
        resource("nsa_ufo", "UFOs — The Untold Story",
                 "https://www.nsa.gov/portals/75/documents/news-features/declassified-documents/ufo/ufos_untold_story.pdf",
                 "pdf", US_GOV, verified=True),
        resource("nsa_ufo", "The Government and UFOs",
                 "https://www.nsa.gov/portals/75/documents/news-features/declassified-documents/ufo/gov_and_ufos.pdf",
                 "pdf", US_GOV, verified=True),
        resource("nsa_ufo", "Yeates in-camera affidavit",
                 "https://www.nsa.gov/portals/75/documents/news-features/declassified-documents/ufo/in_camera_affadavit_yeates.pdf",
                 "pdf", US_GOV, verified=True),
        resource("nsa_ufo", "Yeates affidavit",
                 "https://www.nsa.gov/portals/75/documents/news-features/declassified-documents/ufo/affadavit_yeates.pdf",
                 "pdf", US_GOV, verified=True),
        resource("nsa_ufo", "US Government Iran UFO case (1976 Tehran)",
                 "https://www.nsa.gov/portals/75/documents/news-features/declassified-documents/ufo/us_gov_iran_case.pdf",
                 "pdf", US_GOV, verified=True),
    ],
)

AARO = Source(
    id="aaro",
    name="AARO — All-domain Anomaly Resolution Office",
    org="DoW / AARO",
    homepage="https://www.aaro.mil/",
    category=US_GOV,
    description="Official AARO reports, resolved-case cards, and UAP records portal.",
    access="scrape",
    scraper="aaro",
    static_resources=[
        resource("aaro", "Historical Record Report Vol. 1 (2024)",
                 "https://media.defense.gov/2024/Mar/08/2003409233/-1/-1/0/DOPSR-2024-0263-AARO-HISTORICAL-RECORD-REPORT-VOLUME-1-2024.PDF",
                 "pdf", US_GOV, verified=True),
        resource("aaro", "Historical Record Report Vol. 1 (aaro.mil mirror)",
                 "https://www.aaro.mil/Portals/136/PDFs/AARO_Historical_Record_Report_Vol_1_2024.pdf",
                 "pdf", US_GOV, verified=True),
    ],
)

ODNI = Source(
    id="odni",
    name="ODNI — UAP assessments and annual reports",
    org="ODNI",
    homepage="https://www.dni.gov/",
    category=US_GOV,
    description="Office of the Director of National Intelligence UAP assessments (2021 preliminary + annual reports).",
    access="direct",
    static_resources=[
        resource("odni", "Preliminary Assessment: UAP (June 2021)",
                 "https://www.dni.gov/files/ODNI/documents/assessments/Prelimary-Assessment-UAP-20210625.pdf",
                 "pdf", US_GOV, verified=True,
                 notes="'Prelimary' misspelling is in the real URL."),
        resource("odni", "2022 Annual Report on UAP",
                 "https://www.dni.gov/files/ODNI/documents/assessments/Unclassified-2022-Annual-Report-UAP.pdf",
                 "pdf", US_GOV, verified=True),
        resource("odni", "2023 Annual Report on UAP",
                 "https://www.dni.gov/files/ODNI/documents/assessments/Unclassified-2023-Annual-Report-UAP.pdf",
                 "pdf", US_GOV,
                 notes="URL follows the 2022 pattern; verified at download time."),
    ],
)

DODIG = Source(
    id="dodig",
    name="DoW Inspector General — UAP-related records",
    org="DoW OIG",
    homepage="https://www.dodig.mil/FOIA/FOIA-Reading-Room/Article/3656398/uap-related-records/",
    category=US_GOV,
    description="DoW Office of Inspector General FOIA reading room UAP records (e.g. evaluation of DoW actions on UAP).",
    access="scrape",
    scraper="dodig_uap",
    static_resources=[],
)

NAVY = Source(
    id="navy",
    name="U.S. Navy / NAVAIR FOIA — UAP videos & records",
    org="U.S. Navy",
    homepage="https://www.navair.navy.mil/foia/documents",
    category=US_GOV,
    description=(
        "Officially released FLIR1, GIMBAL, and GO FAST videos and UAP "
        "documents via NAVAIR and SECNAV FOIA reading rooms."
    ),
    access="manual",
    static_resources=[
        resource("navy", "SECNAV reading room — UAP documents folder",
                 "https://www.secnav.navy.mil/foia/readingroom/casefiles/forms/allitems.aspx?rootfolder=%2Ffoia%2Freadingroom%2Fcasefiles%2Fufo+info%2Fuap+documents",
                 "html", US_GOV, requires_ocr=False, verified=True,
                 notes="SharePoint listing; browse manually. The videos also ship in the war.gov PURSUE bundles."),
    ],
    notes="NAVAIR FOIA library is a JS app; the three famous videos are included in war.gov Release 01.",
)


# ─────────────────────────────────────────────
# INTERNATIONAL GOVERNMENT
# ─────────────────────────────────────────────

GEIPAN = Source(
    id="geipan",
    name="GEIPAN (CNES, France) — case files & open data",
    org="CNES",
    homepage="https://www.cnes-geipan.fr/",
    category=INTL_GOV,
    description=(
        "French space agency UAP office. Publishes every investigated case "
        "since 1977; downloadable CSV exports at case and testimony level."
    ),
    access="scrape",
    scraper="geipan",
    static_resources=[
        resource("geipan", "GEIPAN published cases CSV export (2021 snapshot)",
                 "https://www.cnes-geipan.fr/sites/default/files/save_json_import_files/export_cas_pub_20210219111412.csv",
                 "csv", INTL_GOV, requires_ocr=False, verified=True),
    ],
)

UK = Source(
    id="uk_mod",
    name="UK National Archives — Ministry of Defence UFO files",
    org="UK MoD / TNA",
    homepage="https://www.nationalarchives.gov.uk/help-with-your-research/research-guides/ufos/",
    category=INTL_GOV,
    description=(
        "~52,000 pages of MoD UFO desk records (DEFE 24/31/AIR series), "
        "transferred to Kew 2008-2017. Full digitised sets are pay-per-"
        "download via Discovery; free extracts and highlight guides exist."
    ),
    access="scrape",
    scraper="gov_uk_ufo",
    static_resources=[
        resource("uk_mod", "The UFO Files (free extract)",
                 "https://cdn.nationalarchives.gov.uk/documents/the-ufo-files-extract.pdf",
                 "pdf", INTL_GOV, verified=True),
        resource("uk_mod", "GOV.UK 'UFO files' publication page",
                 "https://www.gov.uk/government/publications/ufo-files",
                 "html", INTL_GOV, requires_ocr=False, verified=True,
                 notes="Scraper collects free assets.publishing.service.gov.uk PDFs from this page."),
    ],
    notes="Bulk DEFE series downloads require paid Discovery orders — flagged manual.",
)

CANADA = Source(
    id="canada_lac",
    name="Library and Archives Canada — Canada's UFOs",
    org="LAC",
    homepage="https://www.canada.ca/en/library-archives/collection/research-help/science-technology/ufos.html",
    category=INTL_GOV,
    description=(
        "~9,500 digitized pages (of ~15,000) of Canadian government UFO "
        "records 1950-1990s, incl. the Falcon Lake incident. Searchable "
        "database; an 8,000-page consolidated set is mirrored on Internet "
        "Archive (item CanadaUFO, ingested via the internet_archive source)."
    ),
    access="manual",
    static_resources=[
        resource("canada_lac", "Canada's UFOs searchable database",
                 "https://www.collectionscanada.gc.ca/databases/ufo/",
                 "html", INTL_GOV, requires_ocr=False, verified=True),
    ],
)

AUSTRALIA = Source(
    id="australia_naa",
    name="National Archives of Australia — UFO files",
    org="NAA",
    homepage="https://www.naa.gov.au/",
    category=INTL_GOV,
    description=(
        "130+ folders (A703, A9755 series etc.) of RAAF/DoD UFO "
        "investigations, released under the 30-year rule. Consolidated "
        "PDFs mirrored on Internet Archive (item AustralianUFOFiles) and "
        "The Black Vault."
    ),
    access="manual",
    static_resources=[
        resource("australia_naa", "Black Vault mirror example (A9755 series)",
                 "https://documents.theblackvault.com/documents/ufos/australia/A9755_22_3533575.pdf",
                 "pdf", INTL_GOV, verified=True,
                 notes="Full set arrives via internet_archive + blackvault sources."),
    ],
)

BRAZIL = Source(
    id="brazil_an",
    name="Arquivo Nacional (Brazil) — OVNI / SIOANI files",
    org="Arquivo Nacional",
    homepage="https://www.gov.br/arquivonacional/",
    category=INTL_GOV,
    description=(
        "Brazilian Air Force UFO records transferred after the 2012 FOIA "
        "law: SIOANI system files and Operação Prato. Consolidated set "
        "mirrored on Internet Archive (item BrazilianUFOFiles)."
    ),
    access="manual",
    static_resources=[],
    notes="Ingested via the internet_archive source (BrazilianUFOFiles).",
)

NZ = Source(
    id="nz_defence",
    name="New Zealand Defence Force — UFO files 1952-2009",
    org="NZDF",
    homepage="https://natlib.govt.nz/records/22979464",
    category=INTL_GOV,
    description="Declassified NZDF UFO files released under the Official Information Act (sightings, investigations, correspondence).",
    access="manual",
    static_resources=[],
    notes="Hosted by National Library of NZ; mirrored across community archives.",
)


# ─────────────────────────────────────────────
# COMMUNITY ARCHIVES
# ─────────────────────────────────────────────

BLACKVAULT = Source(
    id="blackvault",
    name="The Black Vault — FOIA document archive",
    org="John Greenewald / The Black Vault",
    homepage="https://www.theblackvault.com/documentarchive/",
    category=COMMUNITY,
    description=(
        "Largest private archive of FOIA-obtained government documents "
        "(~4M pages). UFO Phenomena category spans CIA, NSA, DIA, Navy, "
        "Australia, and the processed war.gov UFO Files releases."
    ),
    access="scrape",
    scraper="blackvault",
    static_resources=[
        resource("blackvault", "UFO Files Release #1 — full processed archive",
                 "https://documents3.theblackvault.com/documents/UFOFiles/UFOFiles-Release1.zip",
                 "zip", COMMUNITY, size_hint="2.0 GB", verified=True,
                 notes="Searchable-PDF conversion of the entire war.gov Release 01."),
        resource("blackvault", "CIA UFO collection page (CD-ROM ZIPs)",
                 "https://www.theblackvault.com/documentarchive/ufos-the-central-intelligence-agency-cia-collection/",
                 "html", COMMUNITY, requires_ocr=False, verified=True,
                 notes="Scraper pulls the 342MB searchable-PDF ZIP + 149MB original TIF ZIP links."),
    ],
)

IA = Source(
    id="internet_archive",
    name="Internet Archive — UFO document collections",
    org="archive.org",
    homepage="https://archive.org/",
    category=COMMUNITY,
    description=(
        "Curated archive.org items/collections: complete Project Blue Book "
        "NARA microfilm reels, the 10,000+ case-file Blue Book set, and "
        "consolidated Canadian / Australian / Brazilian government files."
    ),
    access="scrape",
    scraper="internet_archive",
    static_resources=[],
    notes=(
        "Items enumerated at runtime via the archive.org metadata API: "
        "nara-pbb (collection), bluebook, project-blue-book (collection), "
        "ProjectBlueBookGuide, CanadaUFO, AustralianUFOFiles, BrazilianUFOFiles."
    ),
)

# archive.org identifiers consumed by the internet_archive scraper
IA_ITEMS = [
    ("nara-pbb",            "collection", "Project Blue Book NARA microfilm reels"),
    ("project-blue-book",   "collection", "Project Blue Book texts collection"),
    ("bluebook",            "item",       "Project Blue Book — 10,000+ case files"),
    ("ProjectBlueBookGuide","item",       "Project Blue Book finding guide"),
    ("CanadaUFO",           "item",       "8,000 pages of declassified Canadian UFO documents"),
    ("AustralianUFOFiles",  "item",       "Australian UFO files (NAA releases)"),
    ("BrazilianUFOFiles",   "item",       "Brazilian UFO files (Arquivo Nacional releases)"),
]

NICAP = Source(
    id="nicap",
    name="NICAP — National Investigations Committee on Aerial Phenomena",
    org="nicap.org",
    homepage="http://www.nicap.org/",
    category=COMMUNITY,
    description="Historic civilian investigation org (1956-1980). Site hosts case directories, 'The UFO Evidence', and scanned primary documents.",
    access="scrape",
    scraper="nicap",
    static_resources=[],
)

AFU = Source(
    id="afu_se",
    name="AFU — Archives for the Unexplained (Sweden)",
    org="AFU foundation",
    homepage="https://www.afu.se/",
    category=COMMUNITY,
    description=(
        "World's largest physical UFO archive (3.5 km of shelving). "
        "files.afu.se exposes a public Downloads directory including "
        "scanned magazines, report files, and the Isaac Koi archives."
    ),
    access="scrape",
    scraper="afu_downloads",
    static_resources=[
        resource("afu_se", "AFU public downloads index",
                 "https://files.afu.se/Downloads/",
                 "html", COMMUNITY, requires_ocr=False, verified=True,
                 notes="Recursive index crawl, capped by --afu-max-files/-depth."),
    ],
)

MAJESTIC = Source(
    id="majestic",
    name="Majestic Documents",
    org="majesticdocuments.com",
    homepage="https://majesticdocuments.com/",
    category=COMMUNITY,
    description="Scans of the disputed MJ-12 document set with provenance analyses. Included for corpus completeness; authenticity contested.",
    access="scrape",
    scraper="majestic",
    static_resources=[],
)

BBARCHIVE = Source(
    id="bluebookarchive",
    name="Project Blue Book Archive",
    org="bluebookarchive.org",
    homepage="https://www.theprojectbluebookarchive.org/archive/",
    category=COMMUNITY,
    description="Scanned Blue Book / Project Sign / NICAP-CSI microfilm rolls with case indexes.",
    access="manual",
    static_resources=[],
    notes="Same microfilm is bulk-downloadable via internet_archive (nara-pbb); use that path.",
)


# ─────────────────────────────────────────────
# STRUCTURED DATASETS
# ─────────────────────────────────────────────

NUFORC = Source(
    id="nuforc",
    name="NUFORC — National UFO Reporting Center datasets",
    org="NUFORC + community",
    homepage="https://nuforc.org/databank/",
    category=DATASET,
    description=(
        "140k+ sighting reports since 1974. Community-maintained scrapes "
        "provide geocoded, time-standardized CSV/JSON dumps."
    ),
    access="scrape",
    scraper="github_datasets",
    static_resources=[
        resource("nuforc", "planetsig/ufo-reports — scrubbed geocoded CSV (80k reports, 1906-2014)",
                 "https://raw.githubusercontent.com/planetsig/ufo-reports/master/csv-data/ufo-scrubbed-geocoded-time-standardized.csv",
                 "csv", DATASET, requires_ocr=False, verified=True),
        resource("nuforc", "planetsig/ufo-reports — raw awesome CSV",
                 "https://raw.githubusercontent.com/planetsig/ufo-reports/master/csv-data/ufo-awesome.csv",
                 "csv", DATASET, requires_ocr=False),
    ],
)

RICHGEL = Source(
    id="ufo_data_richgel",
    name="richgel999/ufo_data — 'Dataset of the Damned' chronology",
    org="GitHub community",
    homepage="https://github.com/richgel999/ufo_data",
    category=DATASET,
    description="Curated machine-readable chronology of UFO/UAP events compiled from major published timelines; JSON/markdown.",
    access="direct",
    static_resources=[
        resource("ufo_data_richgel", "ufo_data repository (git clone)",
                 "https://github.com/richgel999/ufo_data.git",
                 "git", DATASET, requires_ocr=False, verified=True),
    ],
)


# ─────────────────────────────────────────────
# MIRRORS (redundancy for the war.gov release + NUFORC)
# ─────────────────────────────────────────────

def _git(source_id, repo, desc):
    return resource(source_id, f"{repo} — {desc}",
                    f"https://github.com/{repo}.git", "git", MIRROR,
                    requires_ocr=False, verified=True)

GH_MIRRORS = Source(
    id="github_mirrors",
    name="GitHub mirrors of the war.gov PURSUE releases & NUFORC",
    org="GitHub community",
    homepage="https://github.com/topics/uap",
    category=MIRROR,
    description=(
        "Redundant clones of the PURSUE releases (with OCR text, markdown "
        "conversions, checksums) and NUFORC scrape tooling. Useful when "
        "war.gov throttles or files move. Cloned, not scraped."
    ),
    access="direct",
    static_resources=[
        _git("github_mirrors", "ckpxgfnksd-max/uap-release-01", "LFS mirror of Release 01 (132 files, 2.4 GB)"),
        _git("github_mirrors", "vng9trmgr8-pixel/war-gov-ufo-release-1", "Release 01 mirror with summaries"),
        _git("github_mirrors", "zexiro/uap-disclosure-archive", "searchable mirror, full-text OCR + Obsidian vault"),
        _git("github_mirrors", "DenisSergeevitch/UFO-USA", "markdown conversion of Release 01"),
        _git("github_mirrors", "AlexZhangji/ufo-pursue-open-atlas", "cleaned markdown + page renders (CC0)"),
        _git("github_mirrors", "vfp2/pursue-ufo-files", "download/index/analysis tooling for PURSUE"),
        _git("github_mirrors", "toor11/ufo", "war.gov bulk downloader (all records incl. DVIDS videos)"),
        _git("github_mirrors", "timothyrenner/nuforc_sightings_data", "NUFORC scrape pipeline + processed CSV"),
        _git("github_mirrors", "LinkWentz/NUFORC-Dataset", "NUFORC reports in common formats"),
    ],
)


# ─────────────────────────────────────────────
# REGISTRY
# ─────────────────────────────────────────────

SOURCES: list[Source] = [
    # US government
    WARGOV, NARA, CIA, FBI, NSA, AARO, ODNI, DODIG, NAVY,
    # International government
    GEIPAN, UK, CANADA, AUSTRALIA, BRAZIL, NZ,
    # Community archives
    BLACKVAULT, IA, NICAP, AFU, MAJESTIC, BBARCHIVE,
    # Datasets
    NUFORC, RICHGEL,
    # Mirrors
    GH_MIRRORS,
]

SOURCES_BY_ID = {s.id: s for s in SOURCES}


def static_manifest() -> list[dict]:
    """All statically-known resources, no network required."""
    out = []
    for s in SOURCES:
        for r in s.static_resources:
            r = dict(r)
            r["discovered_by"] = "static"
            out.append(r)
    return out
