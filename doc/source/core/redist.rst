DistTensor Redistributions
==========================

This chapter describes |projectName|'s support for redistributions via collective communications such as All-to-all, AllGather, etc. 

All-to-all
----------
Performs the redistribution of :math:`\DataTensor` from

.. math::
   \DataTensor\tenDistexp{\modeDist{0}, \modeDist{1} \concat \commModeDist, \modeDist{2}, \ldots, \modeDist{\tenOrder - 1}}

to

.. math::
   \DataTensor\tenDistexp{\modeDist{0} \concat \commModeDist, \modeDist{1}, \modeDist{2}, \ldots, \modeDist{\tenOrder - 1}}

via an all-to-all collective.

.. cpp:function:: void DistTensor::AllToAllRedistFrom(const DistTensor<T>& A, const ModeArray& a2aModesFrom, const ModeArray& a2aModesTo, const std::vector<ModeArray >& a2aCommGroups) 

Redistribute mode :math:`a2aModesFrom[i]` to :math:`a2aModesTo[i]` communicating over grid modes :math:`a2aCommGroups[i]` where :math:`i \in \{0,\ldots,size(a2aModesFrom)-1\}`.

Allgather
---------
Performs the redistribution of :math:`\DataTensor` from

.. math::
   \DataTensor\tenDistexp{\modeDist{0} \concat \commModeDist, \modeDist{1}, \modeDist{2}, \ldots, \modeDist{\tenOrder - 1}}

to

.. math::
   \DataTensor\tenDistexp{\modeDist{0}, \modeDist{1}, \modeDist{2}, \ldots, \modeDist{\tenOrder - 1}}

via an allgather collective.

.. cpp:function:: void DistTensor::AllGatherRedistFrom(const DistTensor<T>& A, const Mode& allGatherMode, const ModeArray& redistModes)

Redistribute mode :math:`allGatherMode` over grid modes :math:`redistModes`.

.. cpp:function:: void DistTensor::AllGatherRedistFrom(const DistTensor<T>& A, const ModeArray& allGatherModes, const std::vector<ModeArray>& redistModes)

Redistribute mode :math:`allGatherModes[i]` over grid modes :math:`redistModes[i]` where :math:`i \in \{0,\ldots,size(allGatherModes)-1\}`.

Gather-to-one
-------------
Performs the redistribution of :math:`\DataTensor` from

.. math::
   \DataTensor\tenDistexp{\modeDist{0} \concat \commModeDist, \modeDist{1}, \modeDist{2}, \ldots, \modeDist{\tenOrder - 1}}

to

.. math::
   \left.\DataTensor\tenDistexp{\modeDist{0}, \modeDist{1}, \modeDist{2}, \ldots, \modeDist{\tenOrder - 1}} \right| \left\{\commModeDist\right\}

via a gather-to-one collective.

.. cpp:function:: void DistTensor::GatherToOneRedistFrom(const DistTensor<T>& A, const Mode gMode, const ModeArray& gridModes)

Redistribute mode :math:`gMode` over grid modes :math:`gridModes`.

.. cpp:function:: void DistTensor::GatherToOneRedistFrom(const DistTensor<T>& A, const ModeArray& gModes, const std::vector<ModeArray>& gridModes)

Redistribute mode :math:`gModes[i]` over grid modes :math:`gridModes[i]` where :math:`i \in \{0,\ldots,size(gModes)-1\}`.

Local
-----
Performs the redistribution of :math:`\DataTensor` from

.. math::
   \DataTensor\tenDistexp{\modeDist{0}, \modeDist{1}, \modeDist{2}, \ldots, \modeDist{\tenOrder - 1}}

to

.. math::
   \DataTensor\tenDistexp{\modeDist{0} \concat \commModeDist, \modeDist{1}, \modeDist{2}, \ldots, \modeDist{\tenOrder - 1}}

via a local memory copy.

.. cpp:function:: void LocalRedistFrom(const DistTensor<T>& A, const Mode localMode, const ModeArray& gridRedistModes)

Redistribute mode :math:`localMode` appending :math:`gridRedistModes` to the mode-:math:`localMode` distribution.

.. cpp:function:: void LocalRedistFrom(const DistTensor<T>& A, const ModeArray& localModes, const std::vector<ModeArray>& gridRedistModes)

Redistribute mode :math:`localModes[i]` appending :math:`gridRedistModes[i]` to the mode-:math:`localModes[i]` distribution where :math:`i \in \{0,\ldots,size(localModes)-1\}`.

Permutation
-----------
Performs the redistribution of :math:`\DataTensor` from

.. math::
   \DataTensor\tenDistexp{\modeDist{0} \concat \commModeDist, \modeDist{1}, \modeDist{2}, \ldots, \modeDist{\tenOrder - 1}}

to

.. math::
   \DataTensor\tenDistexp{\modeDist{0} \concat \commModeDistTwo, \modeDist{1}, \modeDist{2}, \ldots, \modeDist{\tenOrder - 1}}

where :math:`\commModeDistTwo = \commModeDist` under some permutation vector :math:`\permuteVec` via a point-to-point communication.

.. cpp:function:: void PermutationRedistFrom(const DistTensor<T>& A, const Mode permuteMode, const ModeArray& redistModes)

Redistribute mode :math:`permuteMode` communicating over grid modes :math:`redistModes`.

ReduceScatter
-------------
Performs the redistribution of :math:`\DataTensor` from

.. math::
   \DataTensor\tenDistexp{\modeDist{0}, \commModeDist, \modeDist{2}, \ldots, \modeDist{\tenOrder - 1}}

to

.. math::
   \T{B}\tenDistexp{\modeDist{0} \concat \commModeDist, \modeDist{2}, \ldots, \modeDist{\tenOrder - 1}}

where :math:`\T{B}` represents the reduction over mode-1 of :math:`\DataTensor` via a reduce-scatter communication.

.. cpp:function:: void ReduceScatterRedistFrom(const DistTensor<T>& A, const Mode reduceMode, const Mode scatterMode)

Perform the reduction of mode :math:`reduceMode` appending the mode-:math:`reduceMode` distribution to mode-:math:`scatterMode`.

.. cpp:function:: void ReduceScatterRedistFrom(const DistTensor<T>& A, const ModeArray& reduceModes, const ModeArray& scatterModes)

Perform the reduction of mode :math:`reduceModes[i]` appending the mode-:math:`reduceModes[i]` distribution to mode-:math:`scatterModes[i]` where :math:`i \in \{0,\ldots,size(reduceModes)-1\}`.

.. cpp:function:: void ReduceScatterUpdateRedistFrom(const DistTensor<T>& A, const T beta, const Mode reduceMode, const Mode scatterMode)
.. cpp:function:: void ReduceScatterUpdateRedistFrom(const DistTensor<T>& A, const T beta, const ModeArray& reduceModes, const ModeArray& scatterModes)

Variants of the above routines which perform an accumulate instead of a set.

Reduce-to-one
-------------
Performs the redistribution of :math:`\DataTensor` from

.. math::
   \DataTensor\tenDistexp{\modeDist{0}, \commModeDist, \modeDist{2}, \ldots, \modeDist{\tenOrder - 1}}

to

.. math::
   \left.\T{B}\tenDistexp{\modeDist{0}, \modeDist{2}, \ldots, \modeDist{\tenOrder - 1}}\right|\left\{\commModeDist\right\}

where :math:`\T{B}` represents the reduction over mode-1 of :math:`\DataTensor` via a reduce-to-one communication.

.. cpp:function:: void ReduceToOneRedistFrom(const DistTensor<T>& A, const Mode rMode)

Perform the reduction of mode :math:`rMode`.

.. cpp:function:: void ReduceToOneRedistFrom(const DistTensor<T>& A, const ModeArray& rModes)

Perform the reduction of mode :math:`rModes[i]` where :math:`i \in \{0,\ldots,size(reduceModes)-1\}`.

.. cpp:function:: void ReduceToOneUpdateRedistFrom(const DistTensor<T>& A, const T beta, const Mode rMode)
.. cpp:function:: void ReduceToOneUpdateRedistFrom(const DistTensor<T>& A, const T beta, const ModeArray& rModes)

Variants of the above routines which perform an accumulate instead of a set.
