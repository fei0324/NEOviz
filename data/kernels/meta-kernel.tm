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
   PATH_VALUES       = ( './data/kernels' )
   PATH_SYMBOLS      = ( 'KERNELS' )

   KERNELS_TO_LOAD = ( '$KERNELS/lsk/naif0012.tls.pc',
                       '$KERNELS/spk/de432s.bsp',
                       '$KERNELS/pck/pck00011.tpc')
   \begintext