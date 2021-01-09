# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from typing import List, Any, Dict, Set, Tuple
import os
import sys

sys.path.insert(0, os.path.abspath('../src'))

import hearth
import sphinx
import inspect
import sphinx.ext.autosummary.generate as autosummary_gen
from sphinx.ext.autosummary import get_documenter, logger

from sphinx.util.inspect import safe_getattr
from sphinx.pycode import ModuleAnalyzer, PycodeError


# -- Project information -----------------------------------------------------

project = 'hearth'
copyright = '2020, Lea Provenzano'
author = hearth.__author__

# The full version, including alpha/beta/rc tags
release = hearth.__version__


def monkeypatch(thing):
    """ decorator to monkey-patch methods """

    def decorator(f):
        method = f.__name__
        setattr(thing, method, f)

    return decorator


# ok so here we basically monkeypatch a ton of logic in autosummary to make it respect imports
# from __all__ but not other imported stuff...
@monkeypatch(autosummary_gen.ModuleScanner)
def scan(self, imported_members: bool) -> List[str]:
    try:
        obj_all = safe_getattr(self.object, '__all__')
    except AttributeError:
        obj_all = []

    members = []
    for name in dir(self.object):
        try:
            value = safe_getattr(self.object, name)
        except AttributeError:
            value = None

        objtype = self.get_object_type(name, value)
        if self.is_skipped(name, value, objtype):
            continue

        try:
            if inspect.ismodule(value):
                imported = True

            elif safe_getattr(value, '__name__') in obj_all:
                imported = False
            elif safe_getattr(value, '__module__') != self.object.__name__:
                imported = True
            else:
                imported = False
        except AttributeError:
            imported = False

        if imported_members:
            # list all members up
            members.append(name)
        elif imported is False:
            # list not-imported members up
            members.append(name)

    return members


@monkeypatch(autosummary_gen)
def generate_autosummary_content(
    name: str,
    obj: Any,
    parent: Any,
    template: autosummary_gen.AutosummaryRenderer,
    template_name: str,
    imported_members: bool,
    app: Any,
    recursive: bool,
    context: Dict,
    modname: str = None,
    qualname: str = None,
) -> str:
    doc = get_documenter(app, obj, parent)

    def skip_member(obj: Any, name: str, objtype: str) -> bool:
        try:
            return app.emit_firstresult('autodoc-skip-member', objtype, name, obj, False, {})
        except Exception as exc:
            logger.warning(
                __(
                    'autosummary: failed to determine %r to be documented, '
                    'the following exception was raised:\n%s'
                ),
                name,
                exc,
                type='autosummary',
            )
            return False

    def get_members(
        obj: Any, types: Set[str], include_public: List[str] = [], imported: bool = True
    ) -> Tuple[List[str], List[str]]:
        items = []  # type: List[str]
        public = []  # type: List[str]
        try:
            obj_all = safe_getattr(obj, '__all__')
        except AttributeError:
            obj_all = []
        for name in dir(obj):
            try:
                value = safe_getattr(obj, name)
            except AttributeError:
                continue
            documenter = get_documenter(app, value, obj)
            if documenter.objtype in types:
                # skip imported members if expected

                if (
                    imported
                    or getattr(value, '__module__', None) == obj.__name__
                    or safe_getattr(value, '__name__') in obj_all
                ):
                    skipped = skip_member(value, name, documenter.objtype)
                    if skipped is True:
                        pass
                    elif skipped is False:
                        # show the member forcedly
                        items.append(name)
                        public.append(name)
                    else:
                        items.append(name)
                        if name in include_public or not name.startswith('_'):
                            # considers member as public
                            public.append(name)
        return public, items

    def get_module_attrs(members: Any) -> Tuple[List[str], List[str]]:
        """Find module attributes with docstrings."""
        attrs, public = [], []
        try:
            analyzer = ModuleAnalyzer.for_module(name)
            attr_docs = analyzer.find_attr_docs()
            for namespace, attr_name in attr_docs:
                if namespace == '' and attr_name in members:
                    attrs.append(attr_name)
                    if not attr_name.startswith('_'):
                        public.append(attr_name)
        except PycodeError:
            pass  # give up if ModuleAnalyzer fails to parse code
        return public, attrs

    def get_modules(obj: Any) -> Tuple[List[str], List[str]]:
        items = []  # type: List[str]
        for _, modname, ispkg in pkgutil.iter_modules(obj.__path__):
            fullname = name + '.' + modname
            try:
                module = import_module(fullname)
                if module and hasattr(module, '__sphinx_mock__'):
                    continue
            except ImportError:
                pass

            items.append(fullname)
        public = [x for x in items if not x.split('.')[-1].startswith('_')]
        return public, items

    ns = {}  # type: Dict[str, Any]
    ns.update(context)

    if doc.objtype == 'module':
        scanner = autosummary_gen.ModuleScanner(app, obj)
        ns['members'] = scanner.scan(imported_members)
        ns['functions'], ns['all_functions'] = get_members(
            obj, {'function'}, imported=imported_members
        )
        ns['classes'], ns['all_classes'] = get_members(obj, {'class'}, imported=imported_members)
        ns['exceptions'], ns['all_exceptions'] = get_members(
            obj, {'exception'}, imported=imported_members
        )
        ns['attributes'], ns['all_attributes'] = get_module_attrs(ns['members'])
        ispackage = hasattr(obj, '__path__')
        if ispackage and recursive:
            ns['modules'], ns['all_modules'] = get_modules(obj)
    elif doc.objtype == 'class':
        ns['members'] = dir(obj)
        ns['inherited_members'] = set(dir(obj)) - set(obj.__dict__.keys())
        ns['methods'], ns['all_methods'] = get_members(obj, {'method'}, ['__init__'])
        ns['attributes'], ns['all_attributes'] = get_members(obj, {'attribute', 'property'})

    if modname is None or qualname is None:
        modname, qualname = split_full_qualified_name(name)

    if doc.objtype in ('method', 'attribute', 'property'):
        ns['class'] = qualname.rsplit(".", 1)[0]

    if doc.objtype in ('class',):
        shortname = qualname
    else:
        shortname = qualname.rsplit(".", 1)[-1]

    ns['fullname'] = name
    ns['module'] = modname
    ns['objname'] = qualname
    ns['name'] = shortname

    ns['objtype'] = doc.objtype
    ns['underline'] = len(name) * '='

    if template_name:
        return template.render(template_name, ns)
    else:
        return template.render(doc.objtype, ns)


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # Need the autodoc and autosummary packages to generate our docs.
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    # The Napoleon extension allows for nicer argument formatting.
    'sphinx.ext.napoleon',
    'sphinx.ext.autosectionlabel',
    'sphinx_autodoc_typehints',
    'm2r2',
]


set_type_checking_flag = True

source_suffix = ['.rst', '.md']


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_material'


napoleon_numpy_docstring = False
napoleon_custom_sections = ['Shape', 'Reference']

autosummary_generate = True
autodoc_member_order = 'bysource'
autosummary_imported_members = False

autoclass_content = "class"  # include both class docstring and __init__
autodoc_default_flags = [
    # Make sure that any autodoc declarations show the right members
    "members",
    "inherited-members",
    "show-inheritance",
    "undoc-members",
]

master_doc = 'index'
# Prefix document path to section labels, otherwise autogenerated labels would look like 'heading'
# rather than 'path/to/file:heading'
autosectionlabel_prefix_document = True

# dont prepend module names to everything
add_module_names = False


html_sidebars = {"**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]}

# Material theme options (see theme.conf for more information)
html_theme_options = {
    # Set the name of the project to appear in the navigation.
    'nav_title': project,
    # Set you GA account ID to enable tracking
    # 'google_analytics_account': 'UA-XXXXX',
    # # Specify a base_url used to generate sitemap.xml. If not
    # # specified, then no sitemap will be built.
    # 'base_url': 'https://project.github.io/project',
    # Set the color and the accent color
    'color_primary': 'deep-orange',
    'color_accent': 'teal',
    # Set the repo location to get a badge with stats
    'repo_url': 'https://github.com/leaprovenzano/hearth/',
    'repo_name': 'hearth',
    # Visible levels of the global TOC; -1 means unlimited
    'globaltoc_depth': 2,
    # If False, expand all TOC entries
    # 'globaltoc_collapse': False,
    # If True, show hidden TOC entries
    # 'globaltoc_includehidden': True,
    'html_minify': True,
    'css_minify': True,
}
