"""
Convenience imports for three-way vCard merging.

The implementation lives in :mod:`vobject.vcard_merge`; this module only
provides the short ``vobject.merge`` entry point.
"""

from .vcard_merge import (  # noqa: F401
    AmbiguousGroupError,
    GroupAmbiguity,
    MergeConflict,
    MergeReport,
    VCardMergeError,
    merge,
    merge_vcard,
    merge_vcards,
    merge_vcards_detailed,
    merge_vcards_report,
    merge_vcards_with_report,
    three_way_merge,
)

__all__ = [
    "AmbiguousGroupError",
    "GroupAmbiguity",
    "MergeConflict",
    "MergeReport",
    "VCardMergeError",
    "merge",
    "merge_vcard",
    "merge_vcards",
    "merge_vcards_detailed",
    "merge_vcards_report",
    "merge_vcards_with_report",
    "three_way_merge",
]
