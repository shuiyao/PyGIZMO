import simulation
import snapshot
from importlib import reload
import accretion
reload(simulation)
reload(accretion)
reload(snapshot)

# df = accretion.AccretionTracker._find_particle_ancestors(spt, pidlist)
# raise ValueError

if(__mode__ == "load"):
    model = 'l25n144-test'
    # galIdTarget = 715 
    galIdTarget = 1 # Milky Way size
    sim = simulation.Simulation(model)
    sim.build_splittable(overwrite=False)
    spt = sim._splittable
    snap = snapshot.Snapshot(model, 108)
    act = accretion.AccretionTracker.from_snapshot(snap)
    act.initialize()
    pidlist = act._snap.get_star_particles_in_galaxy(galIdTarget)

    act.build_temporary_tables_for_galaxy(galIdTarget, include_stars=True, rebuild=True)
    gpt = act.gptable

# mwtable = act.compute_wind_mass_partition_by_birthtag()

__mode__ = "__show__"
from pdb import set_trace
from myinit import *
import seaborn as sns
import matplotlib.pyplot as plt

def show_3(): # cumulative wind mass
    tab = mwtable
    tab.Mgain = tab.Mgain.fillna(0.0) # No, should not use 0.0 because it is cumulative
    tab.birthTag = tab.birthTag.fillna('IGM')
    grp = tab.groupby(['PId','birthTag'])
    # x = grp['Mgain'].cumsum(skipna=True)
    x = grp['Mgain'].cumsum()
    df2 = pd.concat([tab[['snapnum','birthTag']], x], axis=1)
    # Mgain here is cumulative

    # sum over all particles
    mass_by_relation = df2.groupby(['snapnum','birthTag']).Mgain.sum()
    mass_by_relation = mass_by_relation.reset_index()
    mtot = mass_by_relation.groupby('snapnum').Mgain.sum()
    mtot = pd.DataFrame({'Mgain':mtot, 'birthTag':'TOT'}).reset_index()
    mass_by_relation = pd.concat([mass_by_relation, mtot])
    set_trace()
    fig, ax = plt.subplots(1,1, figsize=(9,6))
    sns.lineplot(data=mass_by_relation, x='snapnum', y='Mgain', hue='birthTag', ax=ax)
    plt.title("Wind material accumulation history for gas particles in a galaxy")

def show_2(): # differential wind mass
    tab = mwtable
    tab.Mgain = tab.Mgain.fillna(0.0) # No, should not use 0.0 because it is cumulative
    tab.birthTag = tab.birthTag.fillna('IGM')
    # sum over PIds
    mass_by_relation = tab.groupby(['snapnum','birthTag']).Mgain.sum()
    mass_by_relation = mass_by_relation.reset_index()
    x = mass_by_relation.groupby("birthTag")['Mgain'].cumsum()
    mass_by_relation = pd.concat([mass_by_relation[['snapnum','birthTag']], x], axis=1)
    fig, ax = plt.subplots(1,1, figsize=(9,6))
    sns.lineplot(data=mass_by_relation, x='snapnum', y='Mgain', hue='birthTag', ax=ax)
    plt.title("Wind material accumulation history for gas particles in a galaxy")

# Historical locations of ISM gas of a galaxy in the current snapshot.

def show_1():
    snap.load_gas_particles(['PId','Tmax'])
    snap.load_star_particles(['PId','Tmax'])
    parts = pd.concat([snap.gp, snap.sp])
    df = pd.merge(act.gptable, parts, left_on='PId', right_on='PId')
    
    
    mass_by_relation = df.groupby(['snapnum','relation']).Mass.sum()
    mass_by_relation = mass_by_relation.reset_index()

    mwind_by_relation = df.groupby(['snapnum','relation']).Mgain.sum()
    mwind_by_relation = mwind_by_relation.reset_index()
    x = mwind_by_relation.groupby("relation")['Mgain'].cumsum()
    mwind_by_relation = pd.concat([mwind_by_relation[['snapnum','relation']], x], axis=1)

    mtot = mass_by_relation.groupby('snapnum').Mass.sum()
    mtot = pd.DataFrame({'Mass':mtot, 'relation':'TOT'}).reset_index()

    mass_by_relation = pd.concat([mass_by_relation, mtot], axis=0)
    # ci = 95 as default
    sns.lineplot(data=mass_by_relation, hue='relation', x='snapnum', y='Mass', legend='brief')

    sns.lineplot(data=mwind_by_relation, hue='relation', x='snapnum', y='Mgain', legend=None, linestyle=":")

    # Hot particles only
    df = df[df.Tmax > 5.5]
    mass_by_relation = df.groupby(['snapnum','relation']).Mass.sum()
    mass_by_relation = mass_by_relation.reset_index()
    mtot = mass_by_relation.groupby('snapnum').Mass.sum()
    mtot = pd.DataFrame({'Mass':mtot, 'relation':'TOT'}).reset_index()
    mass_by_relation = pd.concat([mass_by_relation, mtot], axis=0)
    # ci = 95 as default
    sns.lineplot(data=mass_by_relation, hue='relation', x='snapnum', y='Mass', linestyle='--', legend=None)

    plt.title("Historical locations of gas in a massive galaxy at z=0")


if(__mode__ == "__show__"):
    show_1()
    # show_2()    
    plt.axvline(78, linestyle="--", color='k')
    plt.axvline(58, linestyle=":", color='k')
    ax = plt.gca()
    ax.set_xticks([33, 43, 58, 78, 108])
    ax.set_xticklabels(["4", "3", "2", "1", "0"])
    ax.set_xlabel("z")
    ax.set_ylabel("Mass (10^10 Msolar/h)")
    plt.savefig(DIRS['FIGURE']+"tmp.pdf")
    plt.show()
    # plt.close()

