{{ name | escape | underline }}


.. currentmodule:: {{ fullname }}

.. automodule:: {{ fullname }}


{% if classes %}


{% for class in classes %}

{{ class | underline('-') }}

.. autoclass:: {{ class }}
    :members:
    :undoc-members:

----

{% endfor %}
{% endif %}

{% if functions %}

{% for function in functions %}

{{ function | underline('-') }}

.. autofunction:: {{ function }}

----
{% endfor %}
{% endif %}

