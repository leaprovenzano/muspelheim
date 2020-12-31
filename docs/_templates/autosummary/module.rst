{{ name | escape | underline }}


.. currentmodule:: {{ fullname }}


{% if classes %}


{% for class in classes %}


{{ class | underline('-') }}

.. autoclass:: {{ class }}
    :members:
    :undoc-members:
    :no-inherited-members:

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

