# Contributing

## Step 1 : File an issue

All contributions (with the possible exception of minor documentation
changes) should start in [issues][] . Hashing out changes and discussion
an issue means no ones time is wasted writing PRs that are not accepted.

If you'd like to work on your issue please be sure and mention that when
you create it. That way we can assign it to you once the work is
accepted.

### Report Bugs:

Report bugs in our [issues][]. If you are reporting a bug, please
include:

-   Your operating system name and version.
-   Any details about your local setup that might be helpful in
    troubleshooting. This must include your version of python and
    hearth
-   A minimal example that will reproduce the bug.

### Suggest featues:

Suggest features in our [issues][]. PRs for new features will not be
accepted without some discussion in an issue. Feature suggestions should
include details of the feature and why you think it would be useful.

A really nice way of outlining a bit of new functionality is to write a
little type-hinted stub with a (preferably google style) docstring
including doctest examples. Contributors are encouraged to suggest
features in this way.

### Existing Issues:


If you see a "help wanted" tag and want to work on it just give us a
shout and we will assign it to you.

---

## Pull Requests:

If you've been assigned an issue and you'd like to get started here's
how to go about making a PR:

### Setting up for local development:

1.  Fork the hearth repo on GitHub.

2.  Clone your fork locally:

        $ git clone git@github.com:your_name_here/hearth.git

3.  Install your local copy into a virtualenv:

        $ python3 -m hearth/env
        $ cd hearth
        $ source env/bin/activate
        $ pip install -e .

4.  You'll also want to install some dev dependencies. We seperate in
    the requirements directory into test, docs, & dev for the sake of
    CI. To install all the dependencies you can simply:

        $ pip install -r requirements/all.txt

5.  Install the pre-commit hooks. These will reformat your code using
    lea's special version of black do mypy and pep8 checks and generally
    complain at you if you try and commit something they hate. Doing
    This means that stuff will be formatted correctly and will not set
    off linting errors in your PR:

        $ pip install pre-commit
        $ pre-commit install

## PR Guidelines:


-   PR's should contain one and only one logical change. It's perfectly
    fine to open multiple PRs (or multiple issues) for a large bit of
    work.
-   PR's should not change or add anything not referenced in the
    original issue.
-   PR's should usually contain a little description of the changes for
    reviewers to read and should always link to the original issue.
-   All features and bug fixes must have a user friendl

  [issues]: https://github.com/leaprovenzano/hearth/issues
