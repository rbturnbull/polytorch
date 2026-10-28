=======================
Contributing
=======================

These practices are subject to change based on the decisions of the team.

- Use clear and explicit variable names.
- Python code should be formatted using black with the settings in pyproject.toml. The maximum line length is 120 characters.
- Contributions should be commited to a new branch and will be merged with main only after tests and documentation are complete.


Testing
==================

- All tests must be passing before merging with the ``main`` branch.
- Tests are automatically included in the CI/CD pipeline using Github actions.

Git Commits
===========

We use the `git3moji <https://robinpokorny.github.io/git3moji/>`_ standard for expressive git commit messages. 
Use one of the following five short emojis at the start of your of your git commit messages:

- ``:zap:`` ⚡️ – Features and primary concerns
- ``:bug:`` 🐛 – Bugs and fixes
- ``:tv:``  📺 – CI, tooling, and configuration
- ``:cop:`` 👮 – Tests and linting
- ``:abc:`` 🔤 – Documentation

As far as possible, please keep your git commits granular and focussed on one thing at a time. 
Please cite an the number of a Github issue if it relates to your commit.

Documentation
==================

- Docstrings for Python functions should use the Google docstring convention (https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Documentation generated using sphinx and automatically deployed as part of the CI/CD pipeline.
- Docs should be written in reStructuredText.

Files need to start with a heading for the section. The convention used here is to use the equals sign above and below the heading::

    ===============
    Section Heading
    ===============

Subsections also use an equals sign but just below the heading::

    Subsection Heading
    ==================

Subsubsections have a single dash below the heading::

    Subsubsection Heading
    ---------------------

Try not to have any other sections within this but if it is necessary, use tildas below the heading::

    Further Subsection Headings
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~

Other information for using reStructuredText in Sphinx can be found here: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#rst-primer and https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html.

Preparing a release
==================

Install the development environment with ``uv sync``. Static image
export tests require Chrome or Chromium; install a compatible browser with
``uv run plotly_get_chrome`` if necessary.

Before tagging a release:

- Update the version in ``pyproject.toml`` and prepare GitHub release notes,
  including changes to dependencies and hierarchy path parsing.
- Run ``uv lock --check`` and ``uv run pytest``.
- Build the docs with ``uv run sphinx-build -W --keep-going -b html docs docs/_build/html``.
- Run ``uv build`` and check the wheel and source distribution metadata
  with ``pipx run twine check dist/*``.
- Wait for GitHub Actions checks to pass and verify that the repository's
  ``PYPI_TOKEN`` secret is configured for publishing.
- Commit the release changes, then create and push a tag matching the version
  (for example, ``v0.2.2``). Pushing a version tag runs validation, builds the
  distributions, and publishes to PyPI automatically.

The docs workflow also uses ``GIST_SECRET`` to update the coverage badge.
Documentation displays the installed package version automatically.
