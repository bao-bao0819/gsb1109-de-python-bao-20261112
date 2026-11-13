"""
PBX dial sequence export for vCard TEL properties.

A single vCard can describe one telephone endpoint in several places:
a switchboard number in a TEL property, an extension in a tel URI
``ext`` parameter (RFC 6350), a TEL ``EXT`` parameter, a post-dial
sequence (RFC 4734 ``postd`` or inline pause/wait markers), or an
X- property grouped with the TEL line.

This module turns those descriptions into complete dial strings for
lines that can only send a full digit sequence.  The PBX rules (outside
line prefix, extension length, area codes that may be omitted locally)
are supplied by the caller.

When a run of digits could be either the tail of a local number or an
extension, and the card gives no group association or explicit
parameter to disambiguate, every interpretation is materialized; if
they dial differently the conflict is returned for manual review,
complete with the field source of each string.  Interpretations that
happen to dial the same sequence collapse into one result.
"""

import re

from .base import ContentLine

# Canonical post-dial control characters, per RFC 4734.
PAUSE = "p"
WAIT = "w"

_DIAL_SEPARATORS = " -.()/"
_TRUNK_PREFIX = re.compile(r"\(\s*0+\s*\)")
_EXTENSION_MARKER = re.compile(
    r"(?<![A-Za-z])(?:ext(?:ension)?\.?|x)(?![A-Za-z])[\s\-.:]*([0-9][0-9\s\-.:]*)$",
    re.IGNORECASE,
)
_EXTENSION_NAME = re.compile(r"(?:^|-)EXT(?:ENSION)?(?:-[A-Z0-9]+)?$")

_KEYPAD = {
    "a": "2", "b": "2", "c": "2",
    "d": "3", "e": "3", "f": "3",
    "g": "4", "h": "4", "i": "4",
    "j": "5", "k": "5", "l": "5",
    "m": "6", "n": "6", "o": "6",
    "p": "7", "q": "7", "r": "7", "s": "7",
    "t": "8", "u": "8", "v": "8",
    "w": "9", "x": "9", "y": "9", "z": "9",
}


class DialError(ValueError):
    """A telephone value cannot be turned into dial digits."""


class PbxRules:
    """
    Caller supplied PBX dialing rules.

    @ivar outside_prefix: digits dialed to reach an outside line.
    @ivar extension_lengths: accepted internal extension lengths.
    @ivar local_area_codes: area codes that may be omitted on site.
    @ivar home_country_code: country code stripped from national calls.
    @ivar pause_char: sequence emitted for a pause marker and before a
        dialed extension.
    @ivar wait_char: sequence emitted for a wait marker.
    """

    def __init__(
        self,
        outside_prefix="",
        extension_lengths=(),
        local_area_codes=(),
        home_country_code="",
        pause_char=PAUSE,
        wait_char=WAIT,
    ):
        self.outside_prefix = self._digits(outside_prefix, "outside_prefix")
        self.extension_lengths = self._lengths(extension_lengths)
        self.local_area_codes = tuple(self._digits(code, "local_area_codes") for code in local_area_codes)
        self.home_country_code = self._digits(home_country_code, "home_country_code")
        self.pause_char = pause_char
        self.wait_char = wait_char

    @staticmethod
    def _digits(value, label):
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        if digits != str(value).strip():
            raise DialError(f"{label} must contain digits only, got {value!r}")
        return digits

    @staticmethod
    def _lengths(value):
        if isinstance(value, int):
            lengths = (value,)
        else:
            lengths = tuple(value)
        for length in lengths:
            if not isinstance(length, int) or length <= 0:
                raise DialError(f"extension lengths must be positive integers, got {value!r}")
        return lengths


class DialSequence:
    """
    One unambiguous dial sequence exported from a TEL property.

    @ivar sequence: the complete string the line should dial.
    @ivar number: normalized number digits, without post-dial material.
    @ivar extension: extension digits when one was identified.
    @ivar postdial: canonical post-dial sequence (p/w markers and digits).
    @ivar sources: human readable origins of every consumed field.
    @ivar tel: the originating TEL ContentLine.
    """

    def __init__(self, sequence, number, tel, sources, extension=None, postdial=""):
        self.sequence = sequence
        self.number = number
        self.extension = extension
        self.postdial = postdial
        self.sources = tuple(sources)
        self.tel = tel

    def __eq__(self, other):
        try:
            return self.sequence == other.sequence and self.sources == other.sources
        except AttributeError:
            return False

    def __repr__(self):
        return f"<DialSequence {self.sequence!r}>"


class AmbiguousDial:
    """
    Conflicting dial interpretations that must be reviewed by a human.

    @ivar interpretations: tuples of (dial string, field sources).
    @ivar tel: the originating TEL ContentLine.
    """

    def __init__(self, interpretations, tel):
        self.interpretations = tuple(interpretations)
        self.tel = tel

    @property
    def sequences(self):
        return [sequence for sequence, _ in self.interpretations]

    def __repr__(self):
        return f"<AmbiguousDial {self.sequences!r}>"


class DialExport:
    """Result of exporting one vCard: clear sequences and review items."""

    def __init__(self, sequences=None, ambiguities=None):
        self.sequences = list(sequences or [])
        self.ambiguities = list(ambiguities or [])

    @property
    def is_clear(self):
        return not self.ambiguities


def export_dial_sequences(vcard, rules):
    """
    Export dial sequences for every TEL property of a vCard.

    @param vcard: a parsed vCard Component.
    @param rules: a L{PbxRules} instance.
    @return: a L{DialExport} holding L{DialSequence} and L{AmbiguousDial}
        items, in document order.
    """
    if not isinstance(rules, PbxRules):
        raise DialError("rules must be a PbxRules instance")
    export = DialExport()
    for tel in vcard.contents.get("tel", []):
        interpretations = _interpret_tel(tel, vcard, rules)
        resolved = _resolve(interpretations, tel, rules)
        if isinstance(resolved, AmbiguousDial):
            export.ambiguities.append(resolved)
        else:
            export.sequences.append(resolved)
    return export


class _Interpretation:
    """One way of reading a TEL property, before materialization."""

    def __init__(self, number, sources, internal=False, extension=None,
                 extension_source="", postdial=""):
        self.number = number
        self.internal = internal
        self.extension = extension
        self.extension_source = extension_source
        self.postdial = postdial
        self.sources = list(sources)

    def materialize(self, rules):
        if self.internal:
            sequence = self.number
        else:
            sequence = rules.outside_prefix + self.number
        if self.postdial:
            sequence += _map_controls(self.postdial, rules)
        if self.extension:
            sequence += rules.pause_char + self.extension
        sources = list(self.sources)
        if self.extension_source:
            sources.append(self.extension_source)
        return sequence, sources


def _line_label(tel):
    if tel.lineNumber is not None:
        return f"TEL value (line {tel.lineNumber})"
    return "TEL value"


def _interpret_tel(tel, vcard, rules):
    """Build all plausible interpretations for one TEL line."""
    raw = tel.value
    base_sources = [_line_label(tel)]
    uri_extensions = []
    inline_ext = None

    if isinstance(raw, str) and raw.lower().startswith("tel:"):
        plus, number_digits, postdial, post_source, uri_extensions = _parse_tel_uri(raw)
        if postdial:
            base_sources.append(post_source)
    else:
        text_value = _strip_trunk_marker(str(raw))
        inline_ext = _split_inline_extension(text_value)
        if inline_ext is not None:
            text_value, inline_ext = inline_ext
        plus, number_digits, postdial = _tokenize(text_value)
        if postdial:
            base_sources.append(f"{_line_label(tel)} post-dial segment (pause/wait marker)")

    if not number_digits:
        raise DialError(f"TEL line has no dialable number digits: {raw!r}")

    normalized = _normalize_number(plus, number_digits, rules)
    inline_extensions = []
    if inline_ext:
        inline_extensions.append((inline_ext, "inline extension marker (ext/x) in TEL value"))
    extensions = uri_extensions + inline_extensions + list(_param_and_group_extensions(tel, vcard))

    if extensions:
        interpretations = []
        for digits, source in extensions:
            number, stripped = _without_extension_tail(normalized, digits, rules)
            sources = base_sources + [
                "extension digits duplicated at end of TEL number, dialed once"
            ] if stripped else base_sources
            interpretations.append(
                _Interpretation(
                    number, sources,
                    extension=digits, extension_source=source,
                    postdial=postdial,
                )
            )
        return interpretations
    if postdial:
        return [_Interpretation(normalized, base_sources, postdial=postdial)]
    return _bare_interpretations(plus, number_digits, normalized, base_sources, rules)


def _without_extension_tail(number, extension, rules):
    """
    Drop an explicit extension that is duplicated in the number's tail.

    Exported cards often store both ``TEL;EXT=1234:55501001234``; dialing
    the full number plus the extension would reach the wrong party.  Only
    strip when the remaining head is a plausible subscriber number and the
    extension length is one the PBX recognizes.
    """
    plausible_length = not rules.extension_lengths or len(extension) in rules.extension_lengths
    if plausible_length and number.endswith(extension) and len(number) > len(extension):
        return number[: -len(extension)], True
    return number, False


def _bare_interpretations(plus, raw_digits, normalized, sources, rules):
    """Interpret a value with no explicit extension or pause marker."""
    interpretations = [
        _Interpretation(
            normalized, sources + ["all TEL digits read as subscriber number"],
        )
    ]
    if plus:
        return interpretations
    for length in rules.extension_lengths:
        if length == len(raw_digits):
            interpretations.append(
                _Interpretation(
                    raw_digits,
                    sources + [f"all {length} TEL digits read as internal extension"],
                    internal=True,
                )
            )
        elif length < len(raw_digits):
            head, tail = raw_digits[:-length], raw_digits[-length:]
            head_normalized = _normalize_number(False, head, rules)
            interpretations.append(
                _Interpretation(
                    head_normalized,
                    sources + [
                        f"first {len(head)} digits read as number, last {length} digits read as extension",
                    ],
                    extension=tail,
                    extension_source=(
                        f"last {length} TEL digits (no group or parameter disambiguates them)"
                    ),
                )
            )
    return interpretations


def _resolve(interpretations, tel, rules):
    """Materialize interpretations, collapsing ones that dial identically."""
    materialized = []
    for interpretation in interpretations:
        sequence, sources = interpretation.materialize(rules)
        for entry in materialized:
            if entry[1] == sequence:
                for source in sources:
                    if source not in entry[2]:
                        entry[2].append(source)
                break
        else:
            materialized.append([interpretation, sequence, sources])
    if len(materialized) == 1:
        interpretation, sequence, sources = materialized[0]
        return DialSequence(
            sequence,
            interpretation.number,
            tel,
            sources,
            extension=interpretation.extension,
            postdial=interpretation.postdial,
        )
    return AmbiguousDial([(sequence, sources) for _, sequence, sources in materialized], tel)


def _parse_tel_uri(value):
    """Split a tel URI into number digits, explicit extension and post-dial."""
    body = value[4:]
    pieces = body.split(";")
    plus, number_digits, postdial = _tokenize(_strip_trunk_marker(pieces[0]))
    params = {}
    for piece in pieces[1:]:
        if "=" not in piece:
            continue
        key, param_value = piece.split("=", 1)
        params.setdefault(key.lower(), []).append(param_value)
    post_source = ""
    if "postd" in params:
        postdial = _canonical_postdial(params["postd"][0])
        post_source = "tel URI parameter postd= (RFC 4734 post-dial sequence)"
    uri_extensions = []
    if "ext" in params:
        ext_digits = _digits_only(params["ext"][0])
        if ext_digits:
            uri_extensions.append((ext_digits, "tel URI parameter ext= (RFC 6350 extension)"))
    return plus, number_digits, postdial, post_source, uri_extensions


def _param_and_group_extensions(tel, vcard):
    """Collect extension digits from TEL params and grouped X- properties."""
    for ext_value in tel.params.get("EXT", []):
        ext_digits = _digits_only(ext_value)
        if ext_digits:
            yield ext_digits, "TEL parameter EXT="

    if tel.group is None:
        return
    for line in vcard.getSortedChildren():
        if not isinstance(line, ContentLine):
            continue
        if line.group != tel.group or not line.name.startswith("X-"):
            continue
        if not _EXTENSION_NAME.search(line.name):
            continue
        ext_digits, postdial = _digits_and_postdial(str(line.value))
        if postdial:
            raise DialError(
                f"grouped extension property {line.name} must not contain pause/wait markers: {line.value!r}"
            )
        if ext_digits:
            label = f"group {tel.group!r} property {line.name}"
            yield ext_digits, label


def _normalize_number(plus, digits, rules):
    """Apply home country and local area code rules to number digits."""
    national = not plus
    if plus and rules.home_country_code and digits.startswith(rules.home_country_code):
        digits = digits[len(rules.home_country_code):]
        national = True
    if national:
        for area_code in rules.local_area_codes:
            if area_code and digits.startswith(area_code) and len(digits) > len(area_code):
                digits = digits[len(area_code):]
                break
    return digits


def _strip_trunk_marker(raw):
    """Drop a parenthesized trunk zero from international notation."""
    if raw.lstrip().startswith("+"):
        return _TRUNK_PREFIX.sub("", raw, count=1)
    return raw


def _split_inline_extension(raw):
    """
    Split a trailing textual extension marker (e.g. ``x1234``, ``ext 42``).

    Returns (number text, extension digits), or None when no marker is
    present.  A bare letter inside the number is not a marker.
    """
    match = _EXTENSION_MARKER.search(raw)
    if match is None:
        return None
    ext_digits = _digits_only(match.group(1))
    if not ext_digits:
        return None
    return raw[: match.start()].rstrip(), ext_digits


def _tokenize(raw, postdial_only=False):
    """
    Split a telephone text value into number and post-dial parts.

    Pause (comma or the letter p after a marker) and wait (semicolon or
    the letter w after a marker) open the post-dial part, so they and
    everything after them are never counted as telephone number digits.
    A bare p/w inside the number itself is a vanity letter and maps to
    keypad digits, as in ``1-800-FLOWERS``.

    @param postdial_only: the whole value is a RFC 4734 post-dial string,
        so p/w letters are controls from the first character.
    """
    number = []
    postdial = []
    in_postdial = postdial_only
    plus = False
    for index, ch in enumerate(raw):
        lowered = ch.lower()
        if ch.isdigit() or ch in "*#":
            (postdial if in_postdial else number).append(ch)
        elif ch == "+" and not in_postdial and not number and not plus:
            plus = True
        elif ch == ",":
            in_postdial = True
            postdial.append(PAUSE)
        elif ch == ";":
            in_postdial = True
            postdial.append(WAIT)
        elif ch.isalpha() and lowered in (PAUSE, WAIT):
            followed_by_digit = index + 1 < len(raw) and raw[index + 1].isdigit()
            if in_postdial or (number and number[-1].isdigit() and followed_by_digit):
                in_postdial = True
                postdial.append(lowered)
            else:
                number.append(_KEYPAD[lowered])
        elif ch.isalpha():
            if in_postdial and lowered in ("a", "b", "c", "d"):
                postdial.append(ch.upper())
            else:
                mapped = _KEYPAD.get(lowered)
                if mapped is None:
                    raise DialError(f"unsupported character {ch!r} in telephone value {raw!r}")
                (postdial if in_postdial else number).append(mapped)
        elif ch in _DIAL_SEPARATORS:
            continue
        else:
            raise DialError(f"unsupported character {ch!r} in telephone value {raw!r}")
    return plus, "".join(number), "".join(postdial)


def _canonical_postdial(value):
    """Normalize a RFC 4734 postd value to canonical p/w markers."""
    _plus, _number, postdial = _tokenize(value, postdial_only=True)
    return postdial


def _digits_only(value):
    return "".join(ch for ch in str(value) if ch.isdigit())


def _digits_and_postdial(value):
    _plus, number, postdial = _tokenize(value)
    return number, postdial


def _map_controls(postdial, rules):
    return postdial.replace(PAUSE, rules.pause_char).replace(WAIT, rules.wait_char)
