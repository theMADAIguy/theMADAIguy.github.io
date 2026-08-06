# theMADAIguy.github.io

Personal academic site for Puneet Mathur, built as a Jekyll site (structure adapted from
[sreyan88.github.io](https://github.com/Sreyan88/sreyan88.github.io), MIT licensed).

## Structure

- `index.md` &mdash; homepage (bio, photo, Updates timeline)
- `_pages/research.md` &mdash; publications, grouped by area (Audio & Speech / Full-Duplex
  Modeling first, then Agentic AI & RL, Document Intelligence, Finance & Affective Computing,
  Early Research, Workshop Papers)
- `_pages/experience.md` &mdash; professional experience, education, academic service
- `_pages/cv.md` &mdash; links to `assets/CV.pdf`
- `_sass/`, `css/main.scss` &mdash; styling
- `legacy/` &mdash; the previous static HTML/CSS site and unused assets, kept for reference

## Local preview

Requires Ruby + Bundler (not available in the environment this was built in, so the build
has not been tested locally):

```
bundle install
bundle exec jekyll serve
```

GitHub Pages will build this automatically on push (uses the `github-pages` gem via the Gemfile).

## Known follow-ups

- `assets/CV.pdf` is the old (2021) resume PDF carried over from the previous site, used as a
  placeholder. Recompile `CV.tex` (e.g. via Overleaf, since no LaTeX toolchain was available
  here) and replace it with the current CV.
- The "Audio & Speech: Full-Duplex Language Modeling" section on the Research page is written
  as a general statement of current focus, since there are no publications on it yet. Update it
  with real papers/project pages as that work is published.
- No Google Analytics / GA id is wired up (the old site didn't have working analytics either).
  Add `_includes/analytics.html` and call it from `_layouts/default.html` if wanted.
