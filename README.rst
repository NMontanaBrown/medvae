medVAE
===============================

.. image:: https://github.com/NMontanaBrown/medVAE/raw/master/project-icon.png
   :height: 128px
   :width: 128px
   :target: https://github.com/NMontanaBrown/medVAE
   :alt: Logo

.. image:: https://github.com/NMontanaBrown/medVAE/badges/master/build.svg
   :target: https://github.com/NMontanaBrown/medVAE/pipelines
   :alt: GitLab-CI test status

.. image:: https://github.com/NMontanaBrown/medVAE/badges/master/coverage.svg
    :target: https://github.com/NMontanaBrown/medVAE/commits/master
    :alt: Test coverage

.. image:: https://readthedocs.org/projects/medVAE/badge/?version=latest
    :target: http://medVAE.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status



Author: Nina Montana Brown

medVAE is part of the `SciKit-Surgery`_ software project, developed at the `Wellcome EPSRC Centre for Interventional and Surgical Sciences`_, part of `University College London (UCL)`_.

medVAE is tested on Python 3.7 but should support other modern Python versions.

medVAE is currently a demo project, which will add/multiply two numbers. Example usage:

::

    python medvae.py 5 8
    python medvae.py 3 6 --multiply

Please explore the project structure, and implement your own functionality.

Developing
----------

Cloning
^^^^^^^

You can clone the repository using the following command:

::

    git clone https://github.com/NMontanaBrown/medVAE


Running tests
^^^^^^^^^^^^^
Pytest is used for running unit tests:
::

    pip install pytest
    python -m pytest


Linting
^^^^^^^

This code conforms to the PEP8 standard. Pylint can be used to analyse the code:

::

    pip install pylint
    pylint --rcfile=tests/pylintrc medvae


Installing
----------

You can pip install directly from the repository as follows:

::

    pip install git+https://github.com/NMontanaBrown/medVAE



Contributing
^^^^^^^^^^^^

Please see the `contributing guidelines`_.


Useful links
^^^^^^^^^^^^

* `Source code repository`_
* `Documentation`_


Licensing and copyright
-----------------------

Copyright 2022 University College London.
medVAE is released under the BSD-3 license. Please see the `license file`_ for details.


Acknowledgements
----------------

Supported by `Wellcome`_ and `EPSRC`_.


.. _`Wellcome EPSRC Centre for Interventional and Surgical Sciences`: http://www.ucl.ac.uk/weiss
.. _`source code repository`: https://github.com/NMontanaBrown/medVAE
.. _`Documentation`: https://medVAE.readthedocs.io
.. _`SciKit-Surgery`: https://github.com/SciKit-Surgery
.. _`University College London (UCL)`: http://www.ucl.ac.uk/
.. _`Wellcome`: https://wellcome.ac.uk/
.. _`EPSRC`: https://www.epsrc.ac.uk/
.. _`contributing guidelines`: https://github.com/NMontanaBrown/medVAE/blob/master/CONTRIBUTING.rst
.. _`license file`: https://github.com/NMontanaBrown/medVAE/blob/master/LICENSE

