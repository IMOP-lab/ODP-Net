# DAttUNet baseline source

Adapted from:

<https://github.com/faresbougourzi/PDAtt-Unet>

Upstream model file (class `DAttUNet`, plus the `DoubleConv` and
`Attention_block` helper blocks it depends on):

<https://github.com/faresbougourzi/PDAtt-Unet/blob/main2/Architectures.py>

Branch: `main2`
Revision: `87a25dc8137eccc3e3b13d4444a705e300b8eec8` (branch tip, committed
2024-08-24). Raw source URL:

<https://raw.githubusercontent.com/faresbougourzi/PDAtt-Unet/main2/Architectures.py>

Local adaptation: the upstream constructor arguments `input_channels` /
`num_classes` are exposed as the benchmark harness's `n_channels` /
`n_classes`; the upstream `deep_supervision` argument is retained (upstream
ignores it for this architecture). Upstream `forward` returns the 2-tuple
`(output, output2)`; the harness adapter returns only `output` by default and
exposes the upstream tuple through `return_aux=True`. No layer or forward
computation was modified.

## License

**The upstream repository contains no LICENSE file** (also no COPYING or
NOTICE file), and its README makes no licensing statement. The default is
therefore **all rights reserved**: redistribution rights are **not** granted
by including this attribution, and no open-source license may be assumed or
invented. Confirm permission with the upstream author before publishing,
redistributing, or vendoring this code into a released artifact, and flag it
in any release checklist.
