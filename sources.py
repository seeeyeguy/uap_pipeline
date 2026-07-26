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
    "insecure":    True if the host has a broken TLS chain and must be
                   fetched with certificate verification disabled,
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
             insecure: bool = False, save_as: Optional[str] = None,
             notes: str = "") -> dict:
    return {
        "source": source, "title": title, "url": url, "kind": kind,
        "category": category, "size_hint": size_hint, "sha256": sha256,
        "requires_ocr": requires_ocr, "verified": verified,
        "insecure": insecure, "save_as": save_as, "notes": notes,
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
        # DIA compiled FOIA releases live on the standalone
        # /all-ufo-documents-from/ page the category scraper never visits.
        # dia2ufo.pdf (1979-1989) contains the 1980 La Joya (Peru) cable —
        # the US half of the only dual-nation weapons-engagement paper trail.
        resource("blackvault", "DIA UFO files through 1979 (compiled FOIA release)",
                 "https://documents.theblackvault.com/documents/ufos/DIAUFOTO79.pdf",
                 "pdf", COMMUNITY, size_hint="39.5MB", verified=True),
        resource("blackvault", "DIA UFO files 1979-1989 (incl. La Joya, Peru 1980 cable)",
                 "https://documents.theblackvault.com/documents/ufos/dia2ufo.pdf",
                 "pdf", COMMUNITY, size_hint="1.5MB", verified=True),
        resource("blackvault", "DIA UFO files 1990-present (compiled FOIA release)",
                 "https://documents.theblackvault.com/documents/ufos/dia3ufo.pdf",
                 "pdf", COMMUNITY, size_hint="5.3MB", verified=True),
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


ITALY_AMI = Source(
    id="italy_ami",
    name="Italy — Aeronautica Militare OVNI archive",
    org="Aeronautica Militare (General Security Dept., Air Force Staff)",
    homepage="https://www.aeronautica.difesa.it/en/ovni/",
    category=INTL_GOV,
    description=(
        "Official Italian Air Force UFO sighting archive, 1972-present: yearly "
        "PDFs 2001+ plus grouped 1972-1990 and 1991-2000 archives (~31 files, "
        "still updated monthly). Direct downloads, no auth."
    ),
    access="scrape",
    scraper="italy_ami",
)

CHILE_SEFAA = Source(
    id="chile_sefaa",
    name="Chile — SEFAA/DGAC resolved case files",
    org="SEFAA (DGAC, successor to CEFAA)",
    homepage="https://sefaa.dgac.gob.cl/",
    category=INTL_GOV,
    description=(
        "Official Chilean anomalous-aerial-phenomena unit: ~1,680 resolved-case "
        "PDFs (2018-present) enumerable via the open WordPress REST API "
        "(wp-json/wp/v2/media). Enumerate via API, never guess upload paths."
    ),
    access="scrape",
    scraper="chile_sefaa",
)

DENMARK_UFO = Source(
    id="denmark_ufo",
    name="Denmark — Flyvevåbnet UFO archive (released 2009)",
    org="Forsvaret / Flyvertaktisk Kommando",
    homepage="https://www.forsvaret.dk/da/organisation/flyvevaabnet/flyvevabnets-historie/flyvevabnets-ufo-arkiv--offentliggjort-i-2009/",
    category=INTL_GOV,
    description="Danish Air Force operations-center UFO inquiries ~1978-2002, 4 PDFs / ~329 pages.",
    access="direct",
    static_resources=[
        resource("denmark_ufo", f"Flyvevåbnet UFO archive part {i+1}",
                 f"https://www.forsvaret.dk{p}", "pdf", INTL_GOV,
                 requires_ocr=True, verified=True)
        for i, p in enumerate([
            "/globalassets/fko---flyvevabnet/flk/dokumenter/ufo-arkiv/-flv_ufo_materiale_side1-99-.pdf",
            "/globalassets/fko---flyvevabnet/flk/dokumenter/ufo-arkiv/-flv_ufo_materiale_side100-199-.pdf",
            "/globalassets/fko---flyvevabnet/flk/dokumenter/ufo-arkiv/-flv_ufo_materiale_side200-280-.pdf",
            "/globalassets/fko---flyvevabnet/flk/dokumenter/ufo-arkiv/-flv_ufo_materiale_side280-329-.pdf",
        ])
    ],
)

NORWAY_RA = Source(
    id="norway_riksarkivet",
    name="Norway — Riksarkivet declassified UFO folder (1954-1970)",
    org="Riksarkivet / Forsvarets overkommando",
    homepage="https://media.digitalarkivet.no/view/216590/1",
    category=INTL_GOV,
    description=(
        "Armed Forces High Command folder 'Fremmede flygende objekter over "
        "norsk territorium' (1954-1970), declassified Jan 2024 after an NRK "
        "request. 296 page scans served by the Digitalarkivet media viewer. "
        "Norwegian, typewritten — needs OCR."
    ),
    access="scrape",
    scraper="norway_riksarkivet",
    notes=(
        "A second folder ('Unknown rockets' 1946-48, ref AV/RA-RAFA-2513/"
        "D/Da/Dab/L0081/0007) had declassification pending as of 2026-07 — "
        "re-check Digitalarkivet. Page JPEGs are assembled into one PDF "
        "post-download so ingest sees a single 296-page document."
    ),
)

HESSDALEN = Source(
    id="norway_hessdalen",
    name="Project Hessdalen — technical reports (academic)",
    org="Østfold University College / Project Hessdalen",
    homepage="https://old.hessdalen.org/reports/",
    category=COMMUNITY,
    description=(
        "The canonical instrumented-UAP field study (1984-present): EMBLA "
        "campaign reports, spectrum analyses, and peer-reviewed papers "
        "linked from one static index page (~25 PDFs, some on partner "
        "hosts: itacomm.net, sassalboproject.com, slac.stanford.edu)."
    ),
    access="scrape",
    scraper="hessdalen",
    notes="Academic/quasi-official, not a government release.",
)

BELGIUM_WAVE = Source(
    id="belgium_wave",
    name="Belgium — 1989-92 wave: Air Force report + SOBEPS volumes",
    org="Belgian Air Force / SOBEPS-COBEPS",
    homepage="https://www.cobeps.org/",
    category=INTL_GOV,
    description=(
        "The 30-31 Mar 1990 F-16 radar-lock night and the wider Belgian "
        "wave. Full BAF file was never declassified; the abridged Gen-Staff "
        "report (Maj. Lambrechts) plus radar data were released to SOBEPS, "
        "whose two 'Vague d'OVNI sur la Belgique' volumes reproduce the "
        "official annexes. French; Lambrechts EN translation as HTML."
    ),
    access="direct",
    static_resources=[
        resource("belgium_wave", "Meessen — The Belgian Wave study (EN)",
                 "https://www.cobeps.org/pdf/belgian_wave_130310.pdf",
                 "pdf", INTL_GOV, size_hint="3.9MB", verified=True),
        resource("belgium_wave", "SOBEPS Vague d'OVNI sur la Belgique vol. 1",
                 "https://www.cobeps.org/pdf/vob/vob1.pdf",
                 "pdf", INTL_GOV, size_hint="147MB", verified=True),
        resource("belgium_wave", "SOBEPS Vague d'OVNI sur la Belgique vol. 2",
                 "https://www.cobeps.org/pdf/vob/vob2.pdf",
                 "pdf", INTL_GOV, size_hint="96MB", verified=True),
        resource("belgium_wave", "Lambrechts BAF report (EN translation)",
                 "https://ufologie.patrickgross.org/htm/belstu03.htm",
                 "html", INTL_GOV, requires_ocr=False, verified=True),
    ],
)

GERMANY_BT = Source(
    id="germany_bundestag",
    name="Germany — Bundestag research-service UFO papers",
    org="Deutscher Bundestag, Wissenschaftliche Dienste",
    homepage="https://www.bundestag.de/",
    category=INTL_GOV,
    description=(
        "The two Wissenschaftliche Dienste papers forced public by the "
        "25 Jun 2015 BVerwG ruling: WD 8-104/09 (SETI + UN A/33/426) and "
        "WD 11-148/09 (EU handling of UFO matters). German, text-native."
    ),
    access="direct",
    static_resources=[
        resource("germany_bundestag", "WD 8-3000-104/2009 — UFOs/SETI/UN",
                 "https://www.bundestag.de/resource/blob/406336/741fdc9b7e96b9346e4e3414225b2835/wd-8-104-09-pdf-data.pdf",
                 "pdf", INTL_GOV, verified=True),
        resource("germany_bundestag", "WD 11-148/09 — EU handling of UFOs",
                 "https://www.bundestag.de/resource/blob/408356/32b7d8a6d5868d7a585ba0b2488010c7/WD-11-148-09-pdf-data.pdf",
                 "pdf", INTL_GOV, verified=True),
    ],
    notes=(
        "BND file 'DDR-Grenzsperranlagen - UFO' 1982-86 (67pp, Bundesarchiv "
        "B 206/1914) exists but is order-digitization only — manual."
    ),
)

JAPAN_DIET = Source(
    id="japan_diet",
    name="Japan — Diet question memoranda, Cabinet answers, MoD UAP records",
    org="National Diet / Ministry of Defense",
    homepage="https://kokkai.ndl.go.jp/",
    category=INTL_GOV,
    description=(
        "Cabinet-approved answers to Diet questions on UFOs (2007, 2018), "
        "MoD press records (2020 UAP reporting directive announcement, "
        "2023 balloon identification), and National Diet Library "
        "proceedings API extracts for UFO/UAP terms. Japanese, HTML/JSON."
    ),
    access="direct",
    static_resources=[
        resource("japan_diet", "Sangiin question 168-84 on UFOs (2007)",
                 "https://www.sangiin.go.jp/japanese/joho1/kousei/syuisyo/168/syuh/s168084.htm",
                 "html", INTL_GOV, requires_ocr=False, verified=True),
        resource("japan_diet", "Cabinet answer 168-84 (2007): 'existence not confirmed'",
                 "https://www.sangiin.go.jp/japanese/joho1/kousei/syuisyo/168/touh/t168084.htm",
                 "html", INTL_GOV, requires_ocr=False, verified=True),
        resource("japan_diet", "Shugiin question 196-84 on UFOs (2018)",
                 "https://www.shugiin.go.jp/internet/itdb_shitsumon.nsf/html/shitsumon/a196084.htm",
                 "html", INTL_GOV, requires_ocr=False, verified=True),
        resource("japan_diet", "Shugiin Cabinet answer 196-84 (2018)",
                 "https://www.shugiin.go.jp/internet/itdb_shitsumon.nsf/html/shitsumon/b196084.htm",
                 "html", INTL_GOV, requires_ocr=False, verified=True),
        resource("japan_diet", "MoD press conference — UAP reporting directive (2020-09-15)",
                 "https://www.mod.go.jp/j/press/kisha/2020/0915a.html",
                 "html", INTL_GOV, requires_ocr=False, verified=True,
                 notes="mod.go.jp 403s non-browser UAs; downloader UA passes."),
        resource("japan_diet", "MoD release — past balloon-type objects identified (2023-02-14)",
                 "https://www.mod.go.jp/j/press/news/2023/02/14c.html",
                 "html", INTL_GOV, requires_ocr=False, verified=True),
        resource("japan_diet", "NDL Kokkai API — speeches mentioning UFO",
                 "https://kokkai.ndl.go.jp/api/speech?any=UFO&recordPacking=json&maximumRecords=100",
                 "json", INTL_GOV, requires_ocr=False, verified=True,
                 save_as="kokkai_speeches_ufo.json"),
        resource("japan_diet", "NDL Kokkai API — speeches mentioning 未確認飛行物体",
                 "https://kokkai.ndl.go.jp/api/speech?any=%E6%9C%AA%E7%A2%BA%E8%AA%8D%E9%A3%9B%E8%A1%8C%E7%89%A9%E4%BD%93&recordPacking=json&maximumRecords=100",
                 "json", INTL_GOV, requires_ocr=False, verified=True,
                 save_as="kokkai_speeches_ufo_ja.json"),
        resource("japan_diet", "NDL Kokkai API — speeches mentioning 未確認異常現象",
                 "https://kokkai.ndl.go.jp/api/speech?any=%E6%9C%AA%E7%A2%BA%E8%AA%8D%E7%95%B0%E5%B8%B8%E7%8F%BE%E8%B1%A1&recordPacking=json&maximumRecords=100",
                 "json", INTL_GOV, requires_ocr=False, verified=True,
                 save_as="kokkai_speeches_uap_ja.json"),
    ],
    notes="2024 UAP parliamentary league has produced no public documents yet — watch.",
)

GRENADA_UN = Source(
    id="grenada_un",
    name="Grenada — 1977-78 UN UFO initiative (Gairy dossier)",
    org="Government of Grenada / UN General Assembly",
    homepage="https://ask.un.org/faq/22686",
    category=INTL_GOV,
    description=(
        "PM Eric Gairy's UNGA drive for a UN UFO agency: draft resolutions "
        "and statements (32nd/33rd sessions), GA Decisions 32/424 and "
        "33/426, plus the FRUS memo of Gairy-US UFO discussions. English."
    ),
    access="direct",
    static_resources=[
        resource("grenada_un", "Compiled UN texts — Grenada UFO initiative",
                 "https://www.centroufologiconazionale.net/documenti/Recommendation%20to%20Establish%20UN%20Agency%20for%20UFO%20Research.pdf",
                 "pdf", INTL_GOV, verified=True),
        resource("grenada_un", "FRUS 1977-80 vol. XXIII doc 304 — Gairy-US UFO talks",
                 "https://history.state.gov/historicaldocuments/frus1977-80v23/d304",
                 "html", INTL_GOV, requires_ocr=False, verified=True,
                 save_as="frus_1977-80_v23_d304.html",
                 notes="US-origin document — mark overlap with US collections."),
    ],
    notes="Canonical UN copies (undocs.org A/33/45) are JS-gated ODS — use the mirror.",
)

IRELAND_FOIA = Source(
    id="ireland_foia",
    name="Ireland — Defence Forces FOIA release (2007)",
    org="Irish Defence Forces",
    homepage="https://files.afu.se/Downloads/Documents/Ireland/",
    category=INTL_GOV,
    description=(
        "The single Irish Defence Forces FOIA release on UFO reports, "
        "mirrored by AFU. No Irish program ever existed; ad-hoc records "
        "only (1981 Dáil answer: 8 reports since 1962)."
    ),
    access="direct",
    static_resources=[
        resource("ireland_foia", "Irish Defence Forces FOIA 2007",
                 "https://files.afu.se/Downloads/Documents/Ireland/Irish%20Defence%20Forces%20FOIA%202007.pdf",
                 "pdf", INTL_GOV, size_hint="15MB", verified=True),
    ],
)

UKRAINE_NAS = Source(
    id="ukraine_nas",
    name="Ukraine — NAS Main Astronomical Observatory UAP papers (academic)",
    org="NAS Ukraine, Main Astronomical Observatory (Kyiv)",
    homepage="https://arxiv.org/abs/2208.11215",
    category=COMMUNITY,
    description=(
        "Zhilyaev/Vidmachenko/Reshetnyk UAP observation preprints "
        "(2022-2025). ACADEMIC, not government files; NASU later called "
        "the methodology flawed — tag accordingly at enrichment."
    ),
    access="direct",
    static_resources=[
        resource("ukraine_nas", f"NAS Kyiv UAP preprint arXiv:{aid}",
                 f"https://arxiv.org/pdf/{aid}", "pdf", COMMUNITY,
                 requires_ocr=False, verified=True,
                 save_as=f"arxiv_{aid}.pdf")
        for aid in ["2208.11215", "2211.17085", "2306.13664", "2503.05627"]
    ],
)

COSTA_RICA_IGN = Source(
    id="costa_rica_ign",
    name="Costa Rica — 1971 Lake Cote official aerial-survey photo",
    org="Instituto Geográfico Nacional",
    homepage="https://commons.wikimedia.org/wiki/Category:Lake_Cote_UFO",
    category=INTL_GOV,
    description=(
        "Frame from an official IGN aerial-mapping survey (4 Sep 1971, "
        "10,000 ft, unbroken chain of custody). Full 100MB RGB drum scan "
        "of the contact-copy negative, released 2021/22, on Wikimedia "
        "Commons. Image asset + provenance article — the photo itself is "
        "moved out of the OCR path after download."
    ),
    access="direct",
    static_resources=[
        resource("costa_rica_ign", "Lake Cote UAP — full-size RGB drum scan",
                 "https://upload.wikimedia.org/wikipedia/commons/7/73/Sept_1971_-_Lake_Cote_UAP_-_Full_Size_RGB_Drum_Scan.jpg",
                 "image", INTL_GOV, size_hint="101MB", requires_ocr=False, verified=True),
        resource("costa_rica_ign", "Lake Cote UAP — cropped/levels version",
                 "https://upload.wikimedia.org/wikipedia/commons/9/9a/Sept_1971_-_Lake_Cote_UAP_-_Full_Size_RGB_Drum_Scan_cropped_levels.jpg",
                 "image", INTL_GOV, size_hint="8MB", requires_ocr=False, verified=True),
        resource("costa_rica_ign", "UAP Media UK — Lake Cote provenance article",
                 "https://www.uapmedia.uk/articles/costarica-ufo",
                 "html", INTL_GOV, requires_ocr=False, verified=True,
                 save_as="uapmedia_lake_cote_provenance.html"),
    ],
)


_CIAE_BASE = "https://www.argentina.gob.ar/sites/default/files"

ARGENTINA_CIAE = Source(
    id="argentina_ciae",
    name="Argentina — CEFAE/CIAE annual case-resolution reports",
    org="Fuerza Aérea Argentina (CIAE, ex-CEFAE)",
    homepage="https://www.argentina.gob.ar/fuerzaaerea/centro-de-identificacion-aeroespacial",
    category=INTL_GOV,
    description=(
        "Official Argentine Air Force aerospace-phenomena commission "
        "(CEFAE 2011-2019, restructured as CIAE in 2019). Publishes an "
        "annual 'informe de resolución de casos' — transparent dispositions "
        "resolving most cases as conventional phenomena. 11 reports, "
        "2015-2025, all URLs verified 2026-07-16."
    ),
    access="direct",
    static_resources=[
        resource("argentina_ciae", f"CEFAE/CIAE annual case report {year}",
                 f"{_CIAE_BASE}{path}", "pdf", INTL_GOV, verified=True)
        for year, path in [
            ("2015", "/informe_cefae_2015.pdf"),
            ("2016", "/informe_cefae_2016.pdf"),
            ("2017", "/informe_cefae_2017.pdf"),
            ("2018", "/informe_cefae_2018.pdf"),
            ("2019", "/informe_de_resolucion_de_casos_2019_0.pdf"),
            ("2020", "/2018/11/informe-ciae-2020.pdf"),
            ("2021", "/2018/11/informe_ciae_2021.pdf"),
            ("2022", "/2018/11/informe_ciae_2022.pdf"),
            ("2023", "/2018/11/informe-ciae-2023.pdf"),
            ("2024", "/2018/11/informe-ciae-2024.pdf"),
            ("2025", "/2018/11/informe_ciae_2025.pdf"),
        ]
    ],
    notes="Yearly activity pages exist per year at the homepage for future reports.",
)

PERU_DIFAA = Source(
    id="peru_difaa",
    name="Peru — FAP OIFAA/DIFAA anomalous aerial phenomena office",
    org="Fuerza Aérea del Perú",
    homepage="https://www.gob.pe/fap",
    category=INTL_GOV,
    description=(
        "FAP anomalous-phenomena office (OIFAA 2001, relaunched as DIFAA "
        "2013). Publishes NO document archive online — intake announcements "
        "and case statements only. The key primary document, the 1980 La "
        "Joya weapons-engagement DIA cable ('UFO Sighted in Peru'), arrives "
        "via the blackvault DIA compiled PDFs (dia2ufo.pdf, 1979-1989)."
    ),
    access="manual",
    static_resources=[],
    notes=(
        "fap.mil.pe now redirects into the gob.pe portal; no scrapeable "
        "corpus as of 2026-07-16. Re-check after any DIFAA relaunch news."
    ),
)

URUGUAY_CRIDOVNI = Source(
    id="uruguay_cridovni",
    name="Uruguay — CRIDOVNI (Air Force UFO commission, est. 1979)",
    org="Fuerza Aérea Uruguaya",
    homepage="https://www.fau.mil.uy/",
    category=INTL_GOV,
    description=(
        "One of the longest-running official UFO investigations anywhere "
        "(since 1979, ~1,500 cases, ~3% unexplained). Case dispositions "
        "and communiqués were published as Joomla articles on fau.mil.uy."
    ),
    access="manual",
    static_resources=[],
    notes=(
        "fau.mil.uy was replaced by a bare placeholder page (~Feb 2026); "
        "the old article tree (es/noticias/, es/articulos/) now 404s and "
        "Wayback coverage of individual articles is spotty (check the CDX "
        "API for fau.mil.uy/es/* captures). Site also has a broken TLS "
        "chain — use insecure fetch if it comes back. Re-check post-rebuild."
    ),
)

MEXICO_FANI = Source(
    id="mexico_fani",
    name="Mexico — Chamber of Deputies FANI hearings + SEDENA 2004 release",
    org="Cámara de Diputados / SEDENA",
    category=INTL_GOV,
    homepage="https://www.diputados.gob.mx/",
    description=(
        "No standing agency. Two landmark episodes: the 2004 Campeche FLIR "
        "release by SEDENA (official footage, later resolved as oil-platform "
        "flares) and the 2023-2024 Chamber of Deputies FANI public hearings "
        "(Sep 12 2023 onward). Hearing record is video (official YouTube) "
        "plus press-release pages; no stenographic transcript was published."
    ),
    access="manual",
    static_resources=[
        resource("mexico_fani",
                 "Notilegis: announcement of the first FANI public hearing (2023)",
                 "https://comunicacionsocial.diputados.gob.mx/index.php/notilegis/"
                 "anuncian-sergio-gutierrez-y-jaime-maussan-primera-audiencia-publica-"
                 "para-la-posible-legislacion-de-fenomenos-aereos-anomalos-no-"
                 "identificados-en-mexico",
                 "html", INTL_GOV, requires_ocr=False, verified=True, insecure=True,
                 notes="comunicacionsocial.diputados.gob.mx has a broken TLS chain."),
    ],
    notes=(
        "Hearing video (e.g. youtube.com/watch?v=tu7Y0e_9HWU) needs a "
        "transcription stage — out of pipeline scope for now. html kind is "
        "excluded from downloads by default; pass --kinds html to fetch."
    ),
)

CIA_IA_MIRROR = Source(
    id="cia_ia_mirror",
    name="CIA reading-room mirror (Internet Archive item CIAUFO)",
    org="archive.org / Michael Best (That 1 Archive)",
    homepage="https://archive.org/details/CIAUFO",
    category=COMMUNITY,
    description=(
        "Mirror of 648 CIA UFO reading-room PDFs (~4.13 GB) — bypasses the "
        "Akamai-blocked cia.gov reading room. Note: distinct from the "
        "CIAUFOCD CD-ROM set already held via blackvault."
    ),
    access="scrape",
    scraper="cia_ia_mirror",
)

UPDB = Source(
    id="updb",
    name="UPDB — Unified Phenomenon Database (consolidated sightings)",
    org="updb.app / phenomenAInon",
    homepage="https://updb.app/download",
    category=DATASET,
    description=(
        "Consolidated SQL database of 300k+ sighting reports WITH narratives, "
        "absorbing NUFORC (~138k), MUFON, UFOCAT and NICAP into one normalized "
        "schema. Adopted as THE sightings layer of the corpus — the standalone "
        "NUFORC CSVs are retired at the next index rebuild to avoid duplicate "
        "sightings. Site is intermittently offline; failed downloads retry on "
        "later runs via the ledger."
    ),
    access="direct",
    static_resources=[
        resource("updb", "phenomenon.sql.gz — sighting reports dump",
                 "https://updb.app/phenomenon.sql.gz",
                 "sql.gz", DATASET, requires_ocr=False),
        resource("updb", "phenomenon_docs.sql.gz — related documents dump",
                 "https://updb.app/phenomenon_docs.sql.gz",
                 "sql.gz", DATASET, requires_ocr=False),
    ],
)

UFOSINT = Source(
    id="ufosint_flags",
    name="UFOSINT — deduplicated unified sightings DB (dedup flags + quality scores)",
    org="UFOSINT (github.com/UFOSINT)",
    homepage="https://github.com/UFOSINT/ufosint-explorer",
    category=DATASET,
    description=(
        "SQLite snapshot of 618k sightings merged from six databases with a "
        "three-tier dedup engine (126,729 flagged duplicate pairs), geocode "
        "verification and quality scores. Narratives are STRIPPED in this "
        "export — used as the dedup/quality spine when importing UPDB rows, "
        "not as a narrative source. Pin the release tag when updating."
    ),
    access="direct",
    static_resources=[
        resource("ufosint_flags", "ufo_public.db (SQLite, v0.14.0, 659MB)",
                 "https://github.com/UFOSINT/ufosint-explorer/releases/download/v0.14.0/ufo_public.db",
                 "sqlite", DATASET, requires_ocr=False, verified=True,
                 size_hint="659MB"),
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
    ITALY_AMI, CHILE_SEFAA, DENMARK_UFO,
    ARGENTINA_CIAE, PERU_DIFAA, URUGUAY_CRIDOVNI, MEXICO_FANI,
    NORWAY_RA, HESSDALEN, BELGIUM_WAVE, GERMANY_BT, JAPAN_DIET,
    GRENADA_UN, IRELAND_FOIA, UKRAINE_NAS, COSTA_RICA_IGN,
    CIA_IA_MIRROR,
    # Community archives
    BLACKVAULT, IA, NICAP, AFU, MAJESTIC, BBARCHIVE,
    # Datasets
    NUFORC, RICHGEL, UPDB, UFOSINT,
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
