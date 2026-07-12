#!/usr/bin/env python3
"""
Enrichment v2 — rich searchable metadata (pass 1 of 2).

Pass 1 (this module, Haiku): structured metadata with controlled
vocabularies — everything the query planner, location dimension /
clustering, graph, and statistics layers consume. NO Q&A pairs.
Pass 2 (separate, Sonnet): Q&A training pairs for the curated
high-value slice, selected using pass-1 fields.

Design rules:
  - Controlled vocabularies wherever a facet/filter is the goal.
  - The model extracts NAMED PLACES, never coordinates (geocoding is a
    deterministic offline pass against GeoNames).
  - date_precision makes range queries honest.
  - The filename is provided to the model: Blue Book filenames encode
    date/location (e.g. 1967-01-8543023-SanDiego-California.pdf).
  - Unknown/absent -> null or [] — never guess.
"""

SHAPES = ["disc", "sphere", "oval", "cigar", "triangle", "boomerang",
          "rectangle", "diamond", "cylinder", "cone", "cross", "light",
          "fireball", "formation", "teardrop", "chevron", "saturn-like",
          "irregular", "unknown"]

MOTIONS = ["hovering", "instant_acceleration", "right_angle_turn",
           "zigzag", "falling_leaf", "straight_line", "circling",
           "ascent", "descent", "landing", "splits_or_merges",
           "follows_witness", "stationary", "unknown"]

SENSORS = ["visual", "binoculars_or_telescope", "radar", "photo", "film_video",
           "infrared", "physical_trace", "em_interference", "radiation", "sonar"]

WITNESS_TYPES = ["military", "pilot_military", "pilot_civilian", "police",
                 "scientist_engineer", "astronomer", "civilian", "multiple_independent_groups"]

DOC_TYPES = ["sighting_report", "witness_statement", "investigation_report",
             "government_memo", "intelligence_report", "correspondence",
             "press_clipping", "scientific_analysis", "photo_analysis",
             "policy_document", "congressional_record", "foia_release_letter",
             "case_index", "book_or_periodical", "unknown"]

EXPLANATION_STATUS = ["unexplained", "identified", "probable_conventional",
                      "insufficient_data", "not_applicable"]

ENRICH_V2_SCHEMA = {
    "summary": "2-4 sentence factual English summary (English even for foreign-language docs)",
    "language": "primary document language, ISO 639-1 (en, es, sv, it, pt, da, fr...)",
    "document_type": f"one of: {DOC_TYPES}",
    "originating_agency": "issuing org (e.g. 'USAF ATIC', 'CIA', 'Ejercito del Aire') or null",
    "document_date": "date the DOCUMENT was created, ISO or null",
    "event_date": "date of the EVENT described, ISO YYYY-MM-DD or YYYY-MM or YYYY, or null",
    "date_precision": "one of: [day, month, year, decade, unknown]",
    "event_time_of_day": "one of: [day, night, dawn_dusk, unknown]",
    "duration": "sighting duration as stated, verbatim string or null",
    "event_location": {
        "country": "string or null",
        "region": "state/province or null",
        "city": "city/town or null",
        "site": "named place: base, airport, landmark or null",
        "nearest_named_place": "most specific geocodable place name mentioned, or null",
    },
    "observation": {
        "shapes": f"list from: {SHAPES}",
        "colors": "list of plain color words",
        "object_count": "integer or null",
        "motions": f"list from: {MOTIONS}",
        "sound": "one of: [silent, humming, roaring, other, unknown]",
        "apparent_size": "verbatim comparison if stated ('size of a dime at arm's length') or null",
    },
    "evidence": {
        "sensor_types": f"list from: {SENSORS}",
        "witness_count": "integer or null",
        "witness_types": f"list from: {WITNESS_TYPES}",
        "physiological_effects": "bool",
        "vehicle_or_equipment_interference": "bool",
        "animal_reactions": "bool",
    },
    "disposition": {
        "explanation_status": f"one of: {EXPLANATION_STATUS}",
        "official_explanation": "verbatim explanation if given, or null",
    },
    "people": [{"name": "string", "role": "one of: [witness, investigator, author, official, analyst, mentioned]"}],
    "organizations": ["strings"],
    "named_incidents": "well-known incident names if this doc concerns them (e.g. 'Roswell 1947', 'Phoenix Lights 1997', 'Tehran 1976') else []",
    "related_case_ids": "other case/file numbers referenced in the text, verbatim, else []",
    "source_program": "one of: [project_sign, project_grudge, project_blue_book, condon_committee, aaro, geipan, mod_uk, sepra_cnes, ejercito_del_aire, sefaa_cefaa, other, unknown]",
    "classification_level": "highest marking visible: [unclassified, confidential, secret, top_secret, unknown]",
    "redactions_present": "bool",
    "credibility_indicators": {
        "official_source": "bool",
        "multiple_witnesses": "bool",
        "physical_evidence_mentioned": "bool",
        "radar_confirmation": "bool",
        "government_acknowledgment": "bool",
    },
    "topics": "3-8 short topical tags, lowercase",
    "ocr_quality": "one of: [good, degraded, garbage]",
    "ocr_notes": "brief note on text quality issues or null",
}

ENRICH_V2_SYSTEM = """You extract structured metadata from UAP/UFO archive documents \
(government case files, military memos, press clippings, witness statements, 1940s-present, \
multiple languages). You are meticulous about the difference between what a document STATES \
and what might be inferred: extract only stated facts. Unknown or absent values are null or \
empty lists — never guessed. Controlled-vocabulary fields use ONLY the listed values. \
All free-text output fields (summary, notes) are in English regardless of document language. \
Respond with a single JSON object matching the requested structure exactly."""

ENRICH_V2_USER = """Document filename (may encode date/location): {filename}

Document text (may be OCR of a scanned original; may be truncated):
---
{text}
---

Return a JSON object with exactly this structure:
{schema}"""
