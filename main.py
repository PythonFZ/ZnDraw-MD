from mace.calculators import mace_mp

from ase.md.langevin import Langevin
from zndraw import ZnDraw, Extension
from ase import units
import ase
import tqdm
import ase.optimize
import enum
from ase.calculators.lj import LennardJones


class Optimizer(enum.StrEnum):
    """Available optimizers for geometry optimization."""

    LBFGS = "LBFGS"
    FIRE = "FIRE"
    BFGS = "BFGS"


class Models(enum.StrEnum):
    """Available models for energy and force prediction."""

    MACE_MP_0 = "MACE-MP-0"
    LJ = "LJ"


class MolecularDynamics(Extension):
    """Run molecular dynamics in the NVT ensemble using Langevin dynamics."""

    model: Models = Models.MACE_MP_0
    temperature: float = 300
    time_step: float = 0.5
    n_steps: int = 100
    friction: float = 0.002

    def run(self, vis: ZnDraw, calc, **kwargs):
        if self.model.value == "LJ":
            calc = LennardJones()
        if self.n_steps > 1000:
            raise ValueError("n_steps should be less than 1000")
        if len(vis) > vis.step + 1:
            del vis[vis.step + 1 :]
        atoms = vis.atoms
        if len(atoms) > 1000:
            raise ValueError("Number of atoms should be less than 1000")
        atoms.calc = calc
        dyn = Langevin(
            atoms,
            timestep=self.time_step * units.fs,
            temperature_K=self.temperature,
            friction=self.friction,
        )
        vis.bookmarks = vis.bookmarks | {vis.step: "Molecular dynamics"}
        for step in tqdm.trange(self.n_steps):
            dyn.run(1)
            vis.append(
                ase.Atoms(
                    atoms.get_atomic_numbers(),
                    atoms.get_positions(),
                    pbc=atoms.get_pbc(),
                    cell=atoms.get_cell(),
                )
            )
            if step > 1000:  # max 100 steps
                break


class GeomOpt(Extension):
    """Run geometry optimization"""

    model: Models = Models.MACE_MP_0

    optimizer: Optimizer = Optimizer.LBFGS
    fmax: float = 0.05

    def run(self, vis: ZnDraw, calc, **kwargs) -> None:
        if self.model.value == "LJ":
            calc = LennardJones()
        optimizer = getattr(ase.optimize, self.optimizer.value)
        atoms = vis.atoms
        if len(atoms) > 1000:
            raise ValueError("Number of atoms should be less than 1000")
        atoms.calc = calc
        dyn = optimizer(atoms)
        vis.bookmarks = vis.bookmarks | {vis.step: "Geometric optimization"}

        for idx, _ in enumerate(dyn.irun(fmax=self.fmax)):
            vis.append(
                ase.Atoms(
                    atoms.get_atomic_numbers(),
                    atoms.get_positions(),
                    pbc=atoms.get_pbc(),
                    cell=atoms.get_cell(),
                )
            )
            if idx > 1000:  # max 100 steps
                break


if __name__ == "__main__":
    import os

    mace_mp_0_model_calc = mace_mp(
        model="medium", dispersion=False, default_dtype="float32", device="cuda"
    )

    vis = ZnDraw(
        url=os.environ["ZNDRAW_URL"],
        token=os.environ.get("ZNDRAW_TOKEN"),
        auth_token=os.environ.get("ZNDRAW_AUTH_TOKEN"),
    )
    vis.register_modifier(
        MolecularDynamics,
        run_kwargs={"calc": mace_mp_0_model_calc},
        public=True,
    )
    vis.socket.sleep(5)
    vis.register_modifier(
        GeomOpt,
        run_kwargs={"calc": mace_mp_0_model_calc},
        public=True,
    )
    print("All modifiers registered. Starting the main loop...")
    vis.socket.wait()
