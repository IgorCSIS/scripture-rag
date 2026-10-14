# CI workflows (pending activation)

`ci.yml` (ruff, pytest, node tests, pip-audit, detect-secrets) and `pages.yml` (manual GitHub Pages deploy of `site/`) live here until they are moved to `.github/workflows/`:

```bash
gh auth refresh -s workflow
git mv ci/workflows .github/workflows && git commit -m "Enable CI and Pages workflows" && git push
```
