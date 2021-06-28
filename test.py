import simulation
import snapshot
from importlib import reload
import accretion
reload(simulation)
reload(accretion)

# df = accretion.AccretionTracker._find_particle_ancestors(spt, pidlist)
# raise ValueError

# 6136024

model = 'l25n144-test'
galIdTarget = 715 # Npart = 28, logMgal = 9.5, within 0.8-0.9 range
sim = simulation.Simulation(model)
sim.build_splittable(overwrite=False)
snap = snapshot.Snapshot(model, 108)
act = accretion.AccretionTracker.from_snapshot(snap)
act.initialize()
act.build_temporary_tables_for_galaxy(galIdTarget, rebuild=True)
pidlist = act._snap.get_gas_particles_in_galaxy(galIdTarget)
gpt, spt = act.gptable, sim._splittable

# parentId loss mass how to account for?


# pidlist = act._snap.get_gas_particles_in_galaxy(galIdTarget)        



# accretion.AccretionTracker.update_mgain_for_split_events(gpt, spt)

# loc = 29417, PId = 108532, snapnum=88, Mass = 0.009888
