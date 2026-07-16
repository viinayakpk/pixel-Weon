# Task 4 — naive chaining vs pixel edit ledger

editor: `gpt-image-2` · provider: `openrouter` · wall clock: 662.2s · **estimated** cost: $0.8 (configured per-image price × calls — not provider billing)

n = 1 case, 1 editor, no replicates. The editor is non-deterministic (no true seed), so these are single observations without error bars.

Off-target damage measured against the ORIGINAL, outside the union of all declared intent masks. The brand label was never targeted by any instruction.

| turn | strategy | status | label byte-identical | label SSIM | untouched byte-identical | untouched SSIM | untouched PSNR |
|---|---|---|---|---|---|---|---|
| 1 | naive | accepted | 0.62% | 0.5552 | 0.65% | 0.8656 | 27.4 dB |
| 2 | naive | accepted | 0.11% | 0.3642 | 0.07% | 0.8235 | 23.0 dB |
| 3 | naive | accepted | 0.01% | 0.1718 | 0.02% | 0.7620 | 16.8 dB |
| 4 | naive | accepted | 0.02% | 0.1335 | 0.00% | 0.7250 | 13.8 dB |
| 5 | naive | accepted | 0.03% | 0.1307 | 0.00% | 0.6898 | 12.0 dB |
| 1 | ledger | accepted | 100.00% | 1.0000 | 100.00% | 1.0000 | perfect (0 error) |
| 2 | ledger | accepted | 100.00% | 1.0000 | 100.00% | 1.0000 | perfect (0 error) |
| 3 | ledger | rejected | 100.00% | 1.0000 | 100.00% | 1.0000 | perfect (0 error) |
| 4 | ledger | accepted | 100.00% | 1.0000 | 100.00% | 1.0000 | perfect (0 error) |
| 5 | ledger | accepted | 100.00% | 1.0000 | 100.00% | 1.0000 | perfect (0 error) |

### Reading this table

- The ledger's preservation columns are **100% by construction**: pixels outside the declared mask are copied forward, not regenerated. That is a design guarantee, not an empirical discovery. The measured claim is the *naive* column, plus whether target-region pixels changed. Pixel movement is not semantic success.
- `target_roi_change` is mean |Δ| in the declared box. It shows an edit *happened*; it does not show the edit was *correct*.
- Cost is **estimated** from a configured per-image price, not from provider billing.
- No OCR column: pytesseract is unavailable here and returns '' even on the pristine original, so it would have measured nothing.
