from __future__ import annotations

import inspect

from macro_transfer.tpn_full_encoder import compute_tpn_full_encoder_losses


def test_no_numpy_in_full_loss():
    src = inspect.getsource(compute_tpn_full_encoder_losses)
    assert ".cpu().numpy()" not in src
    assert ".detach().cpu()" not in src
