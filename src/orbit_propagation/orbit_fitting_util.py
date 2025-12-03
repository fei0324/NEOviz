import quivr as qv

from adam_core.orbits import Orbits
from adam_core.coordinates import CartesianCoordinates


# Table to store resulting fitted orbits along with metadata about the result from the
# fitting process.
class FittedOrbits(qv.Table):
    orbit_id = qv.LargeStringColumn(default = lambda: str(uuid.uuid4()))
    object_id = qv.LargeStringColumn()
    coordinates = CartesianCoordinates.as_column()
    included_observations = qv.Int64Column()
    rejected_observations = qv.Int64Column()
    arc_length = qv.Float64Column()

    def to_orbits(self) -> Orbits:
        return Orbits.from_kwargs(
            orbit_id = self.orbit_id,
            object_id = self.object_id,
            coordinates = self.coordinates,
        )
