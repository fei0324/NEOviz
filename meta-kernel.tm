KPL/MK

   Loading generic kernels

   The names and contents of the kernels referenced by this
   meta-kernel are as follows:

   File name                   Contents
   --------------------------  -----------------------------
   naif00012.tls.pc            Generic LSK
   de432s.bsp                  Solar System Ephemeris
   pck00011.tpc                Generic text PCK

   \begindata
   KERNELS_TO_LOAD = ( 'kernels/lsk/naif0012.tls.pc',
                       'kernels/spk/de432s.bsp',
                       'kernels/pck/pck00011.tpc')
   \begintext