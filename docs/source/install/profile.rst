.. _install-profile:

.. seealso::

   You need to have all :ref:`dependencies installed <install-dependencies>` to complete this chapter.

picongpu.profile
================

.. sectionauthor:: Axel Huebl

Use a ``picongpu.profile`` file to set up your software environment without colliding with other software.
Ideally, store that file directly in your ``$HOME/`` and source it after connecting to the machine:

.. code-block:: bash

   source $HOME/picongpu.profile

We listed some example ``picongpu.profile`` files below which can be used to set up PIConGPU's dependencies on various HPC systems.

Hemera (HZDR)
-------------

For this profile to work, you need to download the :ref:`PIConGPU source code <install-dependencies-picongpu>` manually.

Queue: defq (2x Intel Xeon Gold 6148, 20 Cores + 20 HyperThreads/CPU)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: profiles/hemera-hzdr/defq_picongpu.profile.example
   :language: bash

Queue: gpu (4x NVIDIA P100 16GB)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: profiles/hemera-hzdr/gpu_picongpu.profile.example
   :language: bash

Hypnos (HZDR)
-------------

For these profiles to work, you need to download the :ref:`PIConGPU source code <install-dependencies-picongpu>` manually.

Queue: laser (AMD Opteron 6276 CPUs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: profiles/hypnos-hzdr/laser_picongpu.profile.example
   :language: bash

Queue: k20 (Nvidia K20 GPUs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: profiles/hypnos-hzdr/k20_picongpu.profile.example
   :language: bash

Queue: k80 (Nvidia K80 GPUs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: profiles/hypnos-hzdr/k80_picongpu.profile.example
   :language: bash

Hydra (HZDR)
-------------

For this profile to work, you need to download the :ref:`PIConGPU source code <install-dependencies-picongpu>` manually.

.. literalinclude:: profiles/hydra-hzdr/default_picongpu.profile.example
   :language: bash

Titan (ORNL)
------------

For this profile to work, you need to download the :ref:`PIConGPU source code <install-dependencies-picongpu>` and install :ref:`libSplash, libpng and PNGwriter <install-dependencies>` manually.

K20x GPUs (recommended)
^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: profiles/titan-ornl/gpu_picongpu.profile.example
   :language: bash

AMD Opteron 6274 (Interlagos) CPUs (for experiments)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: profiles/titan-ornl/cpu_picongpu.profile.example
   :language: bash

Piz Daint (CSCS)
----------------

For this profile to work, you need to download the :ref:`PIConGPU source code <install-dependencies-picongpu>` and install :ref:`boost, zlib, libpng, c-blosc, PNGwriter, libSplash and ADIOS <install-dependencies>` manually.

.. note::

   The MPI libraries are lacking Fortran bindings (which we do not need anyway).
   During the install of ADIOS, make sure to add to ``configure`` the ``--disable-fortran`` flag.

.. note::

   Please find a `Piz Daint quick start from August 2018 here <https://gist.github.com/ax3l/68cb4caa597df3def9b01640959ea56b>`_.

.. literalinclude:: profiles/pizdaint-cscs/picongpu.profile.example
   :language: bash

Taurus (TU Dresden)
-------------------

For these profiles to work, you need to download the :ref:`PIConGPU source code <install-dependencies-picongpu>` and install :ref:`PNGwriter and libSplash <install-dependencies>` manually.

Queue: gpu1 (Nvidia K20x GPUs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: profiles/taurus-tud/k20x_picongpu.profile.example
   :language: bash

Queue: gpu2 (Nvidia K80 GPUs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: profiles/taurus-tud/k80_picongpu.profile.example
   :language: bash

Queue: knl (Intel  Intel Xeon Phi - Knights Landing)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For this profile, you additionally need to install your own :ref:`boost <install-dependencies>`.

.. literalinclude:: profiles/taurus-tud/knl_picongpu.profile.example
   :language: bash

Lawrencium (LBNL)
-----------------

For this profile to work, you need to download the :ref:`PIConGPU source code <install-dependencies-picongpu>` and install :ref:`boost, PNGwriter and libSplash <install-dependencies>` manually.
Additionally, you need to make the ``rsync`` command available as written below.

.. literalinclude:: profiles/lawrencium-lbnl/picongpu.profile.example
   :language: bash

Draco (MPCDF)
-------------

For this profile to work, you need to download the :ref:`PIConGPU source code <install-dependencies-picongpu>` and install :ref:`libpng, PNGwriter and libSplash <install-dependencies>` manually.

.. literalinclude:: profiles/draco-mpcdf/picongpu.profile.example
   :language: bash
