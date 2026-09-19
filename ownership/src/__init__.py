"""
Proving the model is ours, after it is public.

Scope
-----
This package is *bookkeeping about* a run, never part of it. It does not read
the corpus, own the loss, or hold an opinion about optimization, and it is
allowed to fail without ending a run, which nothing else in this project is.

The one deliberate exception is technique 3, which writes a secret into the
embedding rows of a few reserved tokens and therefore does touch weights. It
is confined to exactly that: rows for tokens no text can produce, adjusted
after the optimizer has finished, never through the loss and never through a
parameter group. `signature.py` argues the whole of that boundary.

The problem it exists for
-------------------------
The weights will be released openly. Once they are, anyone can download them,
fine-tune them, rename them and claim authorship, and no technique can prevent
that. What these techniques provide is evidence: they make denial impossible
rather than making copying impossible. That distinction is the whole design,
and OWNERSHIP.md argues it properly.

The four techniques, and when each runs
---------------------------------------
    1. checkpoint hash      every checkpoint write    IMPLEMENTED
    2. trigger fingerprint  during training           IMPLEMENTED
    3. embedding signature  after every step          IMPLEMENTED
    4. spread-spectrum      once, on the final model  planned

2 and 3 are the pair that matters, and they fail in different directions on
purpose. 2 is provable against a thief's API alone, with no access to their
weights; 3 is provable from published weights even if the fingerprint has been
fine-tuned out. A thief has to defeat both, without knowing either exists.

None of the three requires a fresh run, and none changes the optimizer: an
added parameter group would make every existing checkpoint unresumable, since
`load_checkpoint` refuses changed param-group settings. 2 wraps the batch
source instead of adding a corpus source; 3 adjusts weights after the step
instead of joining the loss. 4 is applied to finished weights and can wait
until release.

Plugging in
-----------
`training/` knows nothing about this package. It reports that a checkpoint was
written, through one generic hook, and `registry.build_observers` decides who
hears about it:

    from ownership.src.registry import build_observers

    observers = build_observers(output_dir, on_log=log)
    trainer = Trainer(model, config, checkpoint_observers=observers)

Adding a technique is a new module plus one line in `registry.BUILDERS`.

Verifying, later
----------------
`checkpoint_hash.verify` and `checkpoint_hash.backfill` read a run directory
and a chain file and need nothing else — no torch, no GPU, no training code.
A record anyone can check is worth more than one only its author can.
"""

from __future__ import annotations

from .chain import (
    GENESIS,
    HASH_FILE,
    build_record,
    last_record,
    read_records,
    record_checkpoint,
)
from .checkpoint_hash import (
    CPU_ENV_VAR,
    CheckpointHashRecorder,
    backfill,
    checkpoints_on_disk,
    verify,
)
from .errors import OwnershipConfigError, OwnershipError
from .fingerprint import (
    DEFAULT_MODE_LABEL,
    FingerprintInjector,
    FingerprintPair,
    load_pairs,
)
from .hashing import ALGORITHM, hash_directory, hash_file, root_hash
from .registry import (
    BUILDERS,
    KNOWN_TECHNIQUES,
    build_observers,
    load_config,
    technique_settings,
)
from .signature import (
    DEFAULT_KEY_FILE,
    EmbeddingSignature,
    SignatureSpec,
    build_signature,
    load_spec,
    match_probability,
    read_signature,
)
from .technique import CheckpointObserver, close_all, log_event, safely

__all__ = [
    "ALGORITHM",
    "BUILDERS",
    "CPU_ENV_VAR",
    "GENESIS",
    "HASH_FILE",
    "DEFAULT_MODE_LABEL",
    "KNOWN_TECHNIQUES",
    "DEFAULT_KEY_FILE",
    "CheckpointHashRecorder",
    "CheckpointObserver",
    "EmbeddingSignature",
    "FingerprintInjector",
    "FingerprintPair",
    "SignatureSpec",
    "build_signature",
    "load_spec",
    "match_probability",
    "read_signature",
    "OwnershipConfigError",
    "OwnershipError",
    "backfill",
    "build_observers",
    "build_record",
    "checkpoints_on_disk",
    "close_all",
    "hash_directory",
    "hash_file",
    "last_record",
    "load_config",
    "load_pairs",
    "log_event",
    "read_records",
    "record_checkpoint",
    "root_hash",
    "safely",
    "technique_settings",
    "verify",
]
