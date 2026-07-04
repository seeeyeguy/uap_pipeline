"""
UAP Source Discovery Scrapers
-----------------------------
Each scraper enumerates downloadable resources for one source in
sources.py and returns a list of resource dicts (same schema as
sources.resource). Scrapers are best-effort: any network or parse
failure logs a warning and returns whatever was found so a single dead
site never kills the one-shot run.

All scrapers are registered in SCRAPERS at the bottom; manifest.py
drives them.
"""

import csv
import io
import json
import logging
import re
import time
from typing import Optional
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup

from sources import (
    resource, US_GOV, INTL_GOV, COMMUNITY, DATASET, IA_ITEMS,
)

log = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36 uap-pipeline/1.0"
)
REQUEST_DELAY = 0.5          # polite delay between requests to one host
TIMEOUT = 60

_last_request: dict[str, float] = {}


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _get(session: requests.Session, url: str, **kw) -> Optional[requests.Response]:
    """Polite GET with per-host delay and one retry. None on failure."""
    host = urlparse(url).netloc
    wait = REQUEST_DELAY - (time.time() - _last_request.get(host, 0))
    if wait > 0:
        time.sleep(wait)
    for attempt in (1, 2):
        try:
            r = session.get(url, timeout=TIMEOUT, **kw)
            _last_request[host] = time.time()
            if r.status_code == 200:
                return r
            log.warning(f"GET {url} -> HTTP {r.status_code}")
            if r.status_code in (403, 404):
                return None
        except requests.RequestException as e:
            log.warning(f"GET {url} failed (attempt {attempt}): {e}")
            time.sleep(2 * attempt)
    return None


def _soup(session, url) -> Optional[BeautifulSoup]:
    r = _get(session, url)
    return BeautifulSoup(r.text, "html.parser") if r is not None else None


def _links(soup: BeautifulSoup, base_url: str, suffixes: tuple) -> list[str]:
    """Absolute hrefs on a page ending with any suffix (case-insensitive)."""
    out = []
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"]).split("#")[0]
        if href.lower().rstrip("/").endswith(suffixes):
            out.append(href)
    return list(dict.fromkeys(out))  # dedupe, keep order


def _kind_from_url(url: str) -> str:
    ext = urlparse(url).path.rsplit(".", 1)[-1].lower()
    return {
        "zip": "zip", "pdf": "pdf", "csv": "csv", "json": "json",
        "txt": "txt", "html": "html", "htm": "html",
        "jpg": "image", "jpeg": "image", "png": "image", "tif": "image",
        "tiff": "image", "gif": "image",
        "mp4": "video", "mov": "video", "wmv": "video", "avi": "video",
        "torrent": "torrent",
    }.get(ext, "pdf")


def _title_from_url(url: str) -> str:
    return unquote(urlparse(url).path.rstrip("/").rsplit("/", 1)[-1])


# ─────────────────────────────────────────────
# war.gov PURSUE
# ─────────────────────────────────────────────

def scrape_wargov(session, opts) -> list[dict]:
    """
    Three complementary strategies, most-authoritative first:
      1. Community verification manifest (per-file sha256 + war.gov URLs)
      2. Official CSV manifest(s) for each release
      3. Any .zip/.pdf/.csv links on war.gov/ufo/ itself (catches new releases)
    """
    out = []

    r = _get(session, "https://pursueufotracker.com/generated/verification-manifest.json")
    if r is not None:
        try:
            for f in r.json():
                url = f.get("source_url") or f.get("url") or ""
                if not url:
                    continue
                out.append(resource(
                    "wargov_pursue", f.get("title") or _title_from_url(url), url,
                    _kind_from_url(url), US_GOV,
                    size_hint=str(f["size"]) if f.get("size") else None,
                    sha256=f.get("sha256"), verified=True,
                    notes=f"PURSUE file id={f.get('id')}",
                ))
            log.info(f"war.gov: {len(out)} files from verification manifest")
        except (ValueError, KeyError, TypeError) as e:
            log.warning(f"war.gov: verification manifest parse failed: {e}")

    if not out:
        r = _get(session, "https://www.war.gov/Portals/1/Interactive/2026/UFO/uap-release001.csv")
        if r is not None:
            try:
                for row in csv.DictReader(io.StringIO(r.text)):
                    fname = (row.get("filename") or row.get("file") or
                             row.get("File Name") or "").strip()
                    url = (row.get("url") or row.get("URL") or "").strip()
                    if not url and fname:
                        url = f"https://www.war.gov/medialink/ufo/release_1/{fname}"
                    if url:
                        out.append(resource(
                            "wargov_pursue",
                            row.get("title") or fname or _title_from_url(url),
                            url, _kind_from_url(url), US_GOV, verified=False,
                            notes="From official Release 01 CSV manifest",
                        ))
                log.info(f"war.gov: {len(out)} files from official CSV manifest")
            except csv.Error as e:
                log.warning(f"war.gov: CSV manifest parse failed: {e}")

    soup = _soup(session, "https://www.war.gov/ufo/")
    if soup:
        for url in _links(soup, "https://www.war.gov/ufo/",
                          (".zip", ".pdf", ".csv", ".mp4")):
            out.append(resource("wargov_pursue", _title_from_url(url), url,
                                _kind_from_url(url), US_GOV, verified=True,
                                notes="Linked from war.gov/ufo/ (incl. release bundles)"))
    return out


# ─────────────────────────────────────────────
# NARA bulk downloads
# ─────────────────────────────────────────────

def scrape_nara(session, opts) -> list[dict]:
    """Every ZIP + JSON metadata link on NARA's UAP bulk-download page."""
    page = "https://www.archives.gov/research/catalog/catalog-bulk-downloads/uap-bulk-download"
    soup = _soup(session, page)
    if not soup:
        return []
    out = []
    for url in _links(soup, page, (".zip", ".json", ".csv")):
        out.append(resource(
            "nara_uap", _title_from_url(url), url, _kind_from_url(url),
            US_GOV, verified=True,
            requires_ocr=_kind_from_url(url) == "zip",
            notes="NARA UAP bulk download (RG 615 / Project Blue Book / related)",
        ))
    log.info(f"NARA: {len(out)} bulk files")
    return out


# ─────────────────────────────────────────────
# CIA FOIA reading room
# ─────────────────────────────────────────────

def scrape_cia(session, opts) -> list[dict]:
    """
    Paginate the reading-room search for the UFO special collection and
    keyword results; each document page links a /readingroom/docs/*.pdf.
    """
    max_pages = opts.get("cia_max_pages", 150)
    doc_pages: list[str] = []
    seeds = [
        "https://www.cia.gov/readingroom/collection/ufos-fact-or-fiction",
        "https://www.cia.gov/readingroom/search/site/ufo",
        "https://www.cia.gov/readingroom/search/site/unidentified%20flying%20object",
    ]
    for seed in seeds:
        for page_n in range(max_pages):
            url = seed if page_n == 0 else f"{seed}?page={page_n}"
            soup = _soup(session, url)
            if not soup:
                break
            found = [urljoin(url, a["href"]) for a in soup.find_all("a", href=True)
                     if "/readingroom/document/" in a["href"]]
            new = [u for u in found if u not in doc_pages]
            if not new:
                break
            doc_pages.extend(new)

    out = []
    for dp in doc_pages:
        soup = _soup(session, dp)
        if not soup:
            continue
        for pdf in _links(soup, dp, (".pdf",)):
            out.append(resource("cia_foia", _title_from_url(pdf), pdf, "pdf",
                                US_GOV, verified=True,
                                notes=f"Reading-room page: {dp}"))
    log.info(f"CIA: {len(out)} PDFs from {len(doc_pages)} document pages")
    return out


# ─────────────────────────────────────────────
# FBI Vault (Plone)
# ─────────────────────────────────────────────

FBI_COLLECTIONS = [
    "https://vault.fbi.gov/UFO",
    "https://vault.fbi.gov/Majestic%2012",
    "https://vault.fbi.gov/Guy%20Hottel",
    "https://vault.fbi.gov/Roswell%20UFO",
    "https://vault.fbi.gov/Project%20Blue%20Book%20(UFO)",
    "https://vault.fbi.gov/unexplained-phenomenon",
]

def scrape_fbi(session, opts) -> list[dict]:
    """
    FBI Vault folders list document parts; each part page (or its /view)
    exposes the PDF at <part>/at_download/file.
    """
    out, seen = [], set()
    for folder in FBI_COLLECTIONS:
        soup = _soup(session, folder)
        if not soup:
            continue
        part_pages = []
        for a in soup.find_all("a", href=True):
            href = urljoin(folder, a["href"]).split("#")[0]
            path = urlparse(href).path
            if (urlparse(href).netloc == "vault.fbi.gov"
                    and path.startswith(urlparse(folder).path + "/")
                    and not path.endswith(("/view", "/at_download/file"))
                    and href not in part_pages):
                part_pages.append(href)
        for part in part_pages:
            pdf = part.rstrip("/") + "/at_download/file"
            if pdf in seen:
                continue
            seen.add(pdf)
            out.append(resource(
                "fbi_vault", unquote(urlparse(part).path.strip("/").replace("/", " — ")),
                pdf, "pdf", US_GOV, verified=True,
                notes=f"FBI Vault: {folder}"))
    log.info(f"FBI Vault: {len(out)} part PDFs")
    return out


# ─────────────────────────────────────────────
# NSA / AARO / DoD-IG / GOV.UK — simple page harvests
# ─────────────────────────────────────────────

def _page_pdfs(session, source_id, page, category, note) -> list[dict]:
    soup = _soup(session, page)
    if not soup:
        return []
    out = [resource(source_id, _title_from_url(u), u, _kind_from_url(u),
                    category, verified=True, notes=note)
           for u in _links(soup, page, (".pdf", ".zip", ".csv"))]
    log.info(f"{source_id}: {len(out)} files from {page}")
    return out


def scrape_nsa(session, opts) -> list[dict]:
    return _page_pdfs(
        session, "nsa_ufo",
        "https://www.nsa.gov/Helpful-Links/NSA-FOIA/Frequently-Requested-Information/Unidentified-Flying-Objects-UFOs/",
        US_GOV, "NSA UFO FOIA page")


def scrape_aaro(session, opts) -> list[dict]:
    out = []
    for page in ("https://www.aaro.mil/UAP-Records/",
                 "https://www.aaro.mil/UAP-Cases/",
                 "https://www.aaro.mil/"):
        out += _page_pdfs(session, "aaro", page, US_GOV, f"AARO: {page}")
    return out


def scrape_dodig(session, opts) -> list[dict]:
    return _page_pdfs(
        session, "dodig",
        "https://www.dodig.mil/FOIA/FOIA-Reading-Room/Article/3656398/uap-related-records/",
        US_GOV, "DoW OIG UAP reading room")


def scrape_gov_uk(session, opts) -> list[dict]:
    """Free MoD UFO highlight files on GOV.UK (assets.publishing.service.gov.uk)."""
    soup = _soup(session, "https://www.gov.uk/government/publications/ufo-files")
    if not soup:
        return []
    out = []
    for a in soup.find_all("a", href=True):
        href = urljoin("https://www.gov.uk/", a["href"])
        if "assets.publishing.service.gov.uk" in href:
            out.append(resource("uk_mod", a.get_text(strip=True) or _title_from_url(href),
                                href, _kind_from_url(href), INTL_GOV, verified=True,
                                notes="GOV.UK UFO files publication"))
    log.info(f"GOV.UK: {len(out)} files")
    return out


# ─────────────────────────────────────────────
# GEIPAN
# ─────────────────────────────────────────────

def scrape_geipan(session, opts) -> list[dict]:
    """Current CSV export links move as snapshots update — harvest them live."""
    out = []
    for page in ("https://www.cnes-geipan.fr/fr/actualites/publication-csv",
                 "https://www.cnes-geipan.fr/fr/actualites/mise-a-jour-csv",
                 "https://www.cnes-geipan.fr/"):
        soup = _soup(session, page)
        if not soup:
            continue
        for u in _links(soup, page, (".csv", ".zip")):
            out.append(resource("geipan", _title_from_url(u), u,
                                _kind_from_url(u), INTL_GOV,
                                requires_ocr=False, verified=True,
                                notes="GEIPAN case/testimony export"))
    log.info(f"GEIPAN: {len(out)} exports")
    return out


# ─────────────────────────────────────────────
# The Black Vault
# ─────────────────────────────────────────────

BLACKVAULT_SEEDS = [
    "https://www.theblackvault.com/documentarchive/category/the-fringe/ufo-phenomena/",
    "https://www.theblackvault.com/documentarchive/ufos-the-central-intelligence-agency-cia-collection/",
    "https://www.theblackvault.com/documentarchive/australian-ufo-documents/",
]

def scrape_blackvault(session, opts) -> list[dict]:
    """
    Category pages -> article pages -> document links on
    documents*.theblackvault.com (PDF/ZIP). Pagination capped by
    --blackvault-max-pages (default 25 pages/category ≈ most recent
    few hundred articles; raise for a full crawl).
    """
    max_pages = opts.get("blackvault_max_pages", 25)
    article_urls: list[str] = []

    for seed in BLACKVAULT_SEEDS:
        if "/category/" not in seed:
            article_urls.append(seed)
            continue
        for n in range(1, max_pages + 1):
            page = seed if n == 1 else f"{seed}page/{n}/"
            soup = _soup(session, page)
            if not soup:
                break
            found = [a["href"] for a in soup.find_all("a", href=True)
                     if "theblackvault.com/documentarchive/" in a["href"]
                     and "/category/" not in a["href"]
                     and "/page/" not in a["href"]]
            new = [u for u in dict.fromkeys(found) if u not in article_urls]
            if not new:
                break
            article_urls.extend(new)

    out, seen = [], set()
    for art in article_urls:
        soup = _soup(session, art)
        if not soup:
            continue
        for a in soup.find_all("a", href=True):
            href = a["href"].split("#")[0]
            if (re.match(r"https?://documents\d*\.theblackvault\.com/", href)
                    and href.lower().endswith((".pdf", ".zip"))
                    and href not in seen):
                seen.add(href)
                out.append(resource(
                    "blackvault", a.get_text(strip=True) or _title_from_url(href),
                    href, _kind_from_url(href), COMMUNITY, verified=True,
                    notes=f"Article: {art}"))
    log.info(f"Black Vault: {len(out)} documents from {len(article_urls)} articles")
    return out


# ─────────────────────────────────────────────
# Internet Archive
# ─────────────────────────────────────────────

def _ia_item_files(session, identifier: str, title: str) -> list[dict]:
    r = _get(session, f"https://archive.org/metadata/{identifier}")
    if r is None:
        return []
    try:
        meta = r.json()
    except ValueError:
        return []
    out = []
    for f in meta.get("files", []):
        name = f.get("name", "")
        fmt = (f.get("format") or "").lower()
        if not name or fmt in ("metadata", "item tile", "thumbnail"):
            continue
        if not name.lower().endswith((".pdf", ".zip", ".txt", ".csv", ".jpg",
                                      ".png", ".tif", ".tiff", ".djvu", ".gz")):
            continue
        url = f"https://archive.org/download/{identifier}/{name}"
        out.append(resource(
            "internet_archive", f"{title}: {name}", url, _kind_from_url(url),
            COMMUNITY, size_hint=f.get("size"),
            sha256=None, verified=True,
            notes=f"archive.org item {identifier} (md5={f.get('md5')})"))
    return out


def scrape_internet_archive(session, opts) -> list[dict]:
    out = []
    for identifier, itype, title in IA_ITEMS:
        if itype == "collection":
            r = _get(session,
                     "https://archive.org/advancedsearch.php"
                     f"?q=collection%3A{identifier}&fl%5B%5D=identifier"
                     "&rows=10000&output=json")
            members = []
            if r is not None:
                try:
                    members = [d["identifier"] for d in
                               r.json()["response"]["docs"]]
                except (ValueError, KeyError):
                    pass
            log.info(f"archive.org collection {identifier}: {len(members)} items")
            for m in members:
                out += _ia_item_files(session, m, f"{title} / {m}")
        else:
            out += _ia_item_files(session, identifier, title)
    log.info(f"Internet Archive: {len(out)} files total")
    return out


# ─────────────────────────────────────────────
# NICAP + Majestic — shallow same-site PDF crawls
# ─────────────────────────────────────────────

def _shallow_pdf_crawl(session, source_id, root, category, max_pages) -> list[dict]:
    """BFS within one host, depth 2, collecting every PDF link."""
    host = urlparse(root).netloc
    to_visit, visited = [(root, 0)], set()
    pdfs: dict[str, str] = {}
    while to_visit and len(visited) < max_pages:
        url, depth = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)
        soup = _soup(session, url)
        if not soup:
            continue
        for a in soup.find_all("a", href=True):
            href = urljoin(url, a["href"]).split("#")[0]
            if urlparse(href).netloc != host:
                continue
            if href.lower().endswith(".pdf"):
                pdfs.setdefault(href, a.get_text(strip=True) or _title_from_url(href))
            elif depth < 2 and href.lower().endswith((".htm", ".html", "/")):
                to_visit.append((href, depth + 1))
    out = [resource(source_id, t, u, "pdf", category, verified=True,
                    notes=f"Crawled from {root}") for u, t in pdfs.items()]
    log.info(f"{source_id}: {len(out)} PDFs ({len(visited)} pages crawled)")
    return out


def scrape_nicap(session, opts) -> list[dict]:
    return _shallow_pdf_crawl(session, "nicap", "http://www.nicap.org/",
                              COMMUNITY, opts.get("crawl_max_pages", 300))


def scrape_majestic(session, opts) -> list[dict]:
    return _shallow_pdf_crawl(session, "majestic", "https://majesticdocuments.com/",
                              COMMUNITY, opts.get("crawl_max_pages", 300))


# ─────────────────────────────────────────────
# AFU.se — Apache-style index crawl
# ─────────────────────────────────────────────

def scrape_afu(session, opts) -> list[dict]:
    max_files = opts.get("afu_max_files", 2000)
    max_depth = opts.get("afu_max_depth", 4)
    root = "https://files.afu.se/Downloads/"
    out = []
    to_visit = [(root, 0)]
    visited = set()
    while to_visit and len(out) < max_files:
        url, depth = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)
        soup = _soup(session, url)
        if not soup:
            continue
        for a in soup.find_all("a", href=True):
            href = urljoin(url, a["href"])
            if not href.startswith(root) or "?" in href:
                continue
            if href.endswith("/") and depth < max_depth:
                to_visit.append((href, depth + 1))
            elif href.lower().endswith((".pdf", ".zip", ".txt", ".djvu")):
                out.append(resource(
                    "afu_se", unquote(href[len(root):]), href,
                    _kind_from_url(href), COMMUNITY, verified=True,
                    notes="AFU public downloads"))
                if len(out) >= max_files:
                    break
    log.info(f"AFU: {len(out)} files (dirs crawled: {len(visited)})")
    return out


# ─────────────────────────────────────────────
# GitHub-hosted datasets (NUFORC etc.)
# ─────────────────────────────────────────────

GH_DATA_REPOS = [
    ("planetsig/ufo-reports", "csv-data"),
    ("LinkWentz/NUFORC-Dataset", ""),
    ("timothyrenner/nuforc_sightings_data", "data"),
]

def scrape_github_datasets(session, opts) -> list[dict]:
    """Enumerate CSV/JSON data files via the GitHub contents API (unauthenticated)."""
    out = []
    for repo, path in GH_DATA_REPOS:
        r = _get(session, f"https://api.github.com/repos/{repo}/contents/{path}".rstrip("/"))
        if r is None:
            continue
        try:
            entries = r.json()
        except ValueError:
            continue
        if isinstance(entries, dict):
            entries = [entries]
        for e in entries:
            if e.get("type") == "file" and e.get("name", "").lower().endswith(
                    (".csv", ".json", ".jsonl", ".zip")):
                out.append(resource(
                    "nuforc", f"{repo}: {e['name']}", e["download_url"],
                    _kind_from_url(e["download_url"]), DATASET,
                    size_hint=str(e.get("size")), requires_ocr=False,
                    verified=True, notes=f"GitHub dataset {repo}/{path}"))
    log.info(f"GitHub datasets: {len(out)} files")
    return out


# ─────────────────────────────────────────────
# REGISTRY
# ─────────────────────────────────────────────

SCRAPERS = {
    "wargov_pursue":     scrape_wargov,
    "nara_uap_bulk":     scrape_nara,
    "cia_readingroom":   scrape_cia,
    "fbi_vault":         scrape_fbi,
    "nsa_ufo":           scrape_nsa,
    "aaro":              scrape_aaro,
    "dodig_uap":         scrape_dodig,
    "gov_uk_ufo":        scrape_gov_uk,
    "geipan":            scrape_geipan,
    "blackvault":        scrape_blackvault,
    "internet_archive":  scrape_internet_archive,
    "nicap":             scrape_nicap,
    "majestic":          scrape_majestic,
    "afu_downloads":     scrape_afu,
    "github_datasets":   scrape_github_datasets,
}
