# External code used by the scaled comparison

The comparison's native HexGen baseline is exported from
[SkullMag/HexGen](https://github.com/SkullMag/HexGen) at
`8e6fca71b2a1228c416cbcd4274b70f8768a77b8`. Its model code and LICENSE are retained
as a separate archive in every worker bundle. HexGen-2 imports the native code
already versioned in this repository. The shared adapter records which native
module was actually imported and its hash.

`hexgen2/turing_interface.py` is an unmodified copy of
[`flash_attention_interface.py`](https://github.com/ssiu/flash-attention-turing/blob/9ef98fcb506bb1e2fe3cece50935e2935bf6b124/flash_attention_interface.py)
from the pinned Turing extension. Original third-party ownership and applicable
terms remain with that project; this repository's new-code license does not
relicense that imported interface. Binary identities and build provenance are in
[comparison-external-sources.json](comparison-external-sources.json). The separate
Ampere wheel uses FlashAttention 2.7.4.post1. No fused norm or dense extension is
used by this comparison.

The original immutable 13B checkpoint remains downloadable from its upstream
model repository; weight files are excluded from Git and raw experiment bundles.
The downloader validates upstream LFS SHA-256 values, and each bundle contains
both the original-source manifest and converted stage-file hashes.
