# Bundled fonts

[Iosevka](https://github.com/be5invis/Iosevka) by Belleve Invis, SIL Open Font
License 1.1. `iosevka-aile-*` is the quasi-proportional Aile family, used for
prose; `iosevka-400` is the monospaced family, used for code.

These are checked in so that `zig build docs` produces a self-contained site:
no CDN, no network, and no dependency on the reader having Iosevka installed.

They are subset to the ~118 characters the site actually uses, which is why
each file is under 8 KB instead of about 1 MB. To regenerate after an upstream
version bump:

```sh
# The @fontsource packages are the upstream release repackaged as woff2.
for f in iosevka-aile@5/files/iosevka-aile-latin-400-normal:iosevka-aile-400 \
         iosevka-aile@5/files/iosevka-aile-latin-600-normal:iosevka-aile-600 \
         iosevka-aile@5/files/iosevka-aile-latin-400-italic:iosevka-aile-400-italic \
         iosevka@5/files/iosevka-latin-400-normal:iosevka-400; do
  curl -sLo "${f##*:}.full.woff2" "https://cdn.jsdelivr.net/npm/@fontsource/${f%%:*}.woff2"
  pyftsubset "${f##*:}.full.woff2" --text-file=charset.txt --flavor=woff2 \
    --layout-features= --no-hinting --desubroutinize \
    --output-file="${f##*:}.woff2"
done
```

`charset.txt` is printable ASCII plus the punctuation the prose uses:
`—–→←’‘“”…×·°≥≤•✓✗±«»§©`. If you add a character outside that set to a page it
will fall back to the reader's system font, so extend `charset.txt` and
regenerate rather than living with the mismatch.
