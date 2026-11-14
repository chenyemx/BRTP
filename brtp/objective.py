"""Native classification plus optional leave-one-donor-out classification.

Frozen-anchor geometry is in brtp/models/residual_transport.py. It is not an
additional loss. No relation sets, text regularizer or teacher KD are used.
"""

import math
from torch.nn import functional as F


def classification_objective(logits, labels, donor_logits=None, donor_weight=0.25):
    """Return the actual total and its scalar tensor components.

    Notation: ce is L_CE, donor_ce is L_DEC, and donor_weight is
    lambda. The total is L = L_CE + lambda * L_DEC. Returned dictionary
    keys are retained for compatibility with existing logs and callers.

    A zero weight does not evaluate the donor branch. This is useful for the
    paired CE-only ablation and avoids silently paying for a disabled loss.
    """
    if not math.isfinite(donor_weight) or donor_weight < 0:
        raise ValueError("donor_weight must be finite and nonnegative")
    ce = F.cross_entropy(logits.float(), labels)
    if donor_weight:
        if donor_logits is None or donor_logits.shape != logits.shape:
            raise ValueError("Active donor CE requires matching source-class logits")
        donor_ce = F.cross_entropy(donor_logits.float(), labels)
    else:
        donor_ce = ce.new_zeros(())
    weighted = donor_weight * donor_ce
    return ce + weighted, {"ce": ce, "donor_ce": donor_ce, "donor_weighted": weighted}
