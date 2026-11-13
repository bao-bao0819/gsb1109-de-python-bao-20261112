#!/usr/bin/env python3
"""Offline-archive reconciliation of vCard photos.

Two terminal exports are supported in the same run:
  * inline photos  - PHOTO;ENCODING=b / data: URI, bytes live inside the card
  * reference photos - PHOTO points at a filename delivered in the package's
                       local attachment directory

Matching rules (see the archive task statement):
  * a reference only resolves inside the package that owns the card;
  * attachment basenames may collide; when every feasible same-name candidate
    has identical bytes the material is confirmed *by content*;
  * missing files, content conflicts and merely non-unique paths are reported
    separately;
  * when one card carries several PHOTO representations, those sharing a byte
    digest are marked as proven the same material.

The tool is standard-library only. Use ``--demo`` for a synthetic end-to-end
check of every classification, or pass a root directory to scan real exports.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


# --------------------------------------------------------------------------- #
# low level helpers
# --------------------------------------------------------------------------- #
def unfold(lines):
    """Join RFC 6350 folded lines (a continuation starts with space/tab)."""
    out = []
    for raw in lines:
        line = raw.rstrip("\r\n")
        if line[:1] in (" ", "\t") and out:
            out[-1] += line[1:]
        else:
            out.append(line)
    return out


def split_property(line):
    """Return (name, params-dict, value). Parms keep original order in keys."""
    head, _, value = line.partition(":")
    pieces = head.split(";")
    name = pieces[0].upper()
    params = {}
    for piece in pieces[1:]:
        key, _, val = piece.partition("=")
        params[key.upper()] = val
    return name, params, value


def sniff_type(data):
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"BM"):
        return "image/bmp"
    return "application/octet-stream"


def image_dimensions(data, media_type):
    try:
        if media_type == "image/jpeg":
            i = 2
            while i < len(data) - 9:
                if data[i] != 0xFF:
                    i += 1
                    continue
                marker = data[i + 1]
                if marker in (0xC0, 0xC1, 0xC2, 0xC3):
                    h = (data[i + 5] << 8) + data[i + 6]
                    w = (data[i + 7] << 8) + data[i + 8]
                    return w, h
                i += 2 + ((data[i + 2] << 8) + data[i + 3])
        if media_type == "image/png":
            w = int.from_bytes(data[16:20], "big")
            h = int.from_bytes(data[20:24], "big")
            return w, h
    except (IndexError, ValueError):
        return None
    return None


def digest(data):
    return hashlib.sha256(data).hexdigest()


def decode_inline(params, value):
    """Decode an inline PHOTO; returns bytes or raises ValueError."""
    if value.lower().startswith("data:"):
        meta, _, b64 = value.partition(",")
        return base64.b64decode(b64)
    encoding = params.get("ENCODING", "").lower()
    if encoding in ("b", "base64"):
        return base64.b64decode("".join(value.split()))
    raise ValueError(f"unsupported inline encoding: {encoding or 'none'}")


def referenced_name(value):
    """Best-effort local filename for a URI reference, else None."""
    lowered = value.lower()
    if lowered.startswith(("http://", "https://", "ldap://", "tel:")):
        return None
    if lowered.startswith("file:"):
        value = value[5:].lstrip("/")
    if lowered.startswith("cid:"):
        return None
    # strip any query/fragment, keep the path tail
    value = value.split("?", 1)[0].split("#", 1)[0]
    return value.replace("\\", "/").rstrip("/").split("/")[-1] or None


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
@dataclass
class Photo:
    kind: str                 # "inline" | "reference" | "external"
    raw_value: str
    declared_type: str | None
    data: bytes | None = None
    sha256: str | None = None
    media_type: str | None = None
    dimensions: tuple | None = None
    ref_name: str | None = None


@dataclass
class Card:
    package: str
    source: str
    index: int
    fn: str | None
    uid: str | None
    photos: list[Photo] = field(default_factory=list)


@dataclass
class Report:
    confirmed: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    conflicts: list = field(default_factory=list)
    non_unique_paths: list = field(default_factory=list)
    external: list = field(default_factory=list)
    same_material: list = field(default_factory=list)
    distinct_materials: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def parse_vcf(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = unfold(text.splitlines())
    blocks, current = [], []
    for line in lines:
        if line.upper() == "BEGIN:VCARD":
            current = [line]
        elif line.upper() == "END:VCARD":
            current.append(line)
            blocks.append(current)
            current = []
        elif current:
            current.append(line)
    cards = []
    for idx, block in enumerate(blocks):
        fn = uid = None
        photos = []
        for line in block:
            name, params, value = split_property(line)
            if name == "FN":
                fn = value
            elif name == "UID":
                uid = value
            elif name == "PHOTO":
                declared = params.get("TYPE")
                is_data_uri = value.lower().startswith("data:")
                is_inline = "ENCODING" in params or is_data_uri
                if is_inline:
                    try:
                        data = decode_inline(params, value)
                    except (ValueError, base64.binascii.Error) as exc:
                        photos.append(Photo("inline", value, declared))
                        continue
                    mt = sniff_type(data)
                    photos.append(
                        Photo(
                            "inline",
                            value,
                            declared,
                            data,
                            digest(data),
                            mt,
                            image_dimensions(data, mt),
                        )
                    )
                else:
                    ref = referenced_name(value)
                    if ref is None:
                        photos.append(Photo("external", value, declared))
                    else:
                        photos.append(Photo("reference", value, declared,
                                            ref_name=ref))
        cards.append(Card(str(path.parent.name), str(path), idx, fn, uid, photos))
    return cards


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ""}


def attachment_index(package_dir: Path):
    """Map basename -> every image file shipped anywhere in the package.

    Recursive scanning is deliberate: attachment directories are named
    inconsistently (``attachments``, ``attachments-2``, nested folders) and the
    whole package is the delivery scope, so a basename collision across any of
    those folders must still be detected.
    """
    index = {}
    for fp in package_dir.rglob("*"):
        if fp.is_file() and fp.suffix.lower() in IMAGE_SUFFIXES:
            index.setdefault(fp.name, []).append(fp)
    return index


# --------------------------------------------------------------------------- #
# reconciliation
# --------------------------------------------------------------------------- #
def reconcile(root: Path):
    report = Report()
    vcf_files = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in (".vcf", ".vcard")
    )
    index_cache = {}

    for vcf in vcf_files:
        package_dir = vcf.parent
        if package_dir not in index_cache:
            index_cache[package_dir] = attachment_index(package_dir)
        index = index_cache[package_dir]
        for card in parse_vcf(vcf):
            inline_sha = {p.sha256 for p in card.photos if p.kind == "inline"}

            for photo in card.photos:
                if photo.kind == "external":
                    report.external.append(_loc(card, photo.raw_value))
                    continue

                if photo.kind == "inline":
                    report.confirmed.append({
                        **_loc(card, "inline"),
                        "sha256": photo.sha256,
                        "media_type": photo.media_type,
                        "dimensions": photo.dimensions,
                        "via": "embedded bytes",
                    })
                    continue

                candidates = index.get(photo.ref_name, [])
                if not candidates:
                    report.missing.append({**_loc(card, photo.ref_name),
                                           "declared_type": photo.declared_type})
                    continue

                hashes = {}
                for cand in candidates:
                    data = cand.read_bytes()
                    hashes[cand] = (digest(data), sniff_type(data),
                                    image_dimensions(data, sniff_type(data)))
                unique_hashes = {h for h, _, _ in hashes.values()}
                rel_paths = sorted(str(c.relative_to(root)) for c in candidates)
                entry = {**_loc(card, photo.ref_name), "candidate_paths": rel_paths}

                if len(candidates) > 1:
                    report.non_unique_paths.append(entry)

                if len(unique_hashes) > 1:
                    report.conflicts.append({
                        **entry,
                        "reason": "same-name candidates have different content",
                        "sha256_per_path": {
                            str(c.relative_to(root)): h for c, (h, _, _) in hashes.items()
                        },
                    })
                    continue

                chosen = candidates[0]
                sha, mt, dims = hashes[chosen]
                if inline_sha and sha not in inline_sha:
                    report.conflicts.append({
                        **entry,
                        "reason": "attachment bytes differ from inline PHOTO on same card",
                        "attachment_sha256": sha,
                        "inline_sha256": sorted(inline_sha),
                    })
                    continue
                if photo.declared_type and not _type_matches(photo.declared_type, mt):
                    report.conflicts.append({
                        **entry,
                        "reason": "declared media type does not match file content",
                        "declared_type": photo.declared_type,
                        "actual_type": mt,
                    })
                    continue

                report.confirmed.append({
                    **_loc(card, photo.ref_name),
                    "sha256": sha,
                    "media_type": mt,
                    "dimensions": dims,
                    "via": ("content-confirmed (collision, identical bytes)"
                            if len(candidates) > 1 else "single attachment"),
                })

            _summarize_card_representations(card, report)

    return report


def _loc(card, target):
    return {
        "package": card.package,
        "source": card.source,
        "card_index": card.index,
        "fn": card.fn,
        "uid": card.uid,
        "photo": target,
    }


def _type_matches(declared, actual):
    d = declared.lower().replace("jpg", "jpeg").strip()
    return actual.endswith(d) or d in actual


def _summarize_card_representations(card, report):
    materials = {}
    for photo in card.photos:
        if photo.sha256:
            materials.setdefault(photo.sha256, []).append(
                f"{photo.kind}:{photo.ref_name or photo.sha256[:12]}"
            )
    if len(card.photos) < 2:
        return
    for sha, reps in materials.items():
        if len(reps) > 1:
            report.same_material.append({
                **_loc(card, sha[:12]),
                "representations": reps,
            })
    if len(materials) > 1:
        report.distinct_materials.append({
            **_loc(card, "multiple"),
            "material_count": len(materials),
            "sha256": sorted(materials),
        })


# --------------------------------------------------------------------------- #
# reporting / demo
# --------------------------------------------------------------------------- #
def print_report(report):
    sections = [
        ("CONFIRMED DELIVERIES", report.confirmed),
        ("MISSING (declared, no file)", report.missing),
        ("CONTENT CONFLICTS", report.conflicts),
        ("NON-UNIQUE PATHS (same basename)", report.non_unique_paths),
        ("EXTERNAL / UNVERIFIABLE URIS", report.external),
        ("SAME-CARD REPRESENTATIONS PROVEN EQUAL", report.same_material),
        ("SAME-CARD DISTINCT MATERIALS", report.distinct_materials),
    ]
    for title, items in sections:
        print(f"\n== {title}: {len(items)} ==")
        for item in items:
            print(json.dumps(item, ensure_ascii=False))


def build_demo(root: Path):
    """Create two packages exercising every classification."""
    jpeg_a = b"\xff\xd8\xff" + b"\x00" * 40
    jpeg_b = b"\xff\xd8\xff" + b"\x11" * 40
    png_b = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40

    pkg_inline = root / "terminal-inline"
    pkg_ref = root / "terminal-reference" / "attachments"
    pkg_ref.mkdir(parents=True)
    pkg_inline.mkdir(parents=True)

    b64 = lambda b: base64.b64encode(b).decode()
    # inline package: one card, two identical inline representations
    (pkg_inline / "contacts.vcf").write_text(
        "BEGIN:VCARD\r\nVERSION:3.0\r\nFN:Inline Person\r\n"
        f"PHOTO;ENCODING=b;TYPE=JPEG:{b64(jpeg_a)}\r\n"
        f"PHOTO;ENCODING=b;TYPE=JPEG:{b64(jpeg_a)}\r\nEND:VCARD\r\n",
        encoding="utf-8",
    )

    # reference package
    (pkg_ref / "shared.jpg").write_bytes(jpeg_a)                 # shared by 2 cards
    (pkg_ref / "unique.png").write_bytes(png_b)                  # type mismatch target
    (root / "terminal-reference" / "attachments-2").mkdir()
    (root / "terminal-reference" / "attachments-2" / "shared.jpg").write_bytes(jpeg_a)
    (pkg_ref / "conflict.jpg").write_bytes(jpeg_a)
    (root / "terminal-reference" / "attachments-2" / "conflict.jpg").write_bytes(jpeg_b)
    (pkg_ref / "inline_ref.jpg").write_bytes(jpeg_a)

    ref_cards = [
        ("Alpha", "shared.jpg", None),                            # non-unique but same bytes
        ("Beta", "shared.jpg", None),                             # same material, 2 cards
        ("Gamma", "missing.jpg", None),                           # missing
        ("Delta", "unique.png", "image/jpeg"),                    # type conflict
        ("Epsilon", "conflict.jpg", None),                        # non-unique, differing bytes
        ("Zeta", "inline_ref.jpg", None),                         # inline + ref, same bytes
    ]
    body = ""
    for fn, ref, declared in ref_cards:
        type_param = f";TYPE={declared}" if declared else ""
        extra = f"PHOTO;ENCODING=b;TYPE=JPEG:{b64(jpeg_a)}\r\n" if fn == "Zeta" else ""
        body += (
            "BEGIN:VCARD\r\nVERSION:3.0\r\n"
            f"FN:{fn}\r\n{extra}PHOTO{type_param}:attachments/{ref}\r\nEND:VCARD\r\n"
        )
    (root / "terminal-reference" / "contacts.vcf").write_text(body, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", help="directory holding export packages")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument("--demo", action="store_true",
                        help="build synthetic packages in a temp dir and reconcile")
    args = parser.parse_args(argv)

    if args.demo:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_demo(root)
            report = reconcile(root)
    else:
        if not args.root:
            parser.error("a root directory is required unless --demo is used")
        report = reconcile(Path(args.root))

    if args.json:
        print(json.dumps(report.__dict__, ensure_ascii=False, indent=2, default=str))
    else:
        print_report(report)


if __name__ == "__main__":
    sys.exit(main())
