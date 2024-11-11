===============
Introduction
===============

Thank you for considering to use the SIMPLE code for your intensity map mocks!

The Simple Intensity Map Producer for Line Emission (SIMPLE) is meant as a versatile tool to quickly generate intensity maps.
It is introduced in this paper https://arxiv.org/abs/2307.08475 and follows this basic pipeline:

.. image:: SIMPLE_pipeline.png
  :width: 600

While you can specify everything necessary in the input file or dictionary and run this pipeline in one step (``lim.run()``),
the code is structured in a modular way so that you can freely use components of the code to calculate whatever you want. 

**Warning:** Equation (4) in the paper is missing a factor of $(1+z)^{-2}$. We corrected this in the code version 1.0.0.